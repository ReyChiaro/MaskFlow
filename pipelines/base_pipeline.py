import copy
import torch
import dataclasses

from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from loguru import logger
from PIL import Image
from typing import Any

from utils.summary import summarize_model
from pipelines.utils import get_nested_attr, set_nested_attr
from trainer.parallel.handler import parallel_handler
from trainer.parallel.fsdp_strategy import FSDPStrategy


@dataclasses.dataclass
class BasePipeline:
    r"""
    Base pipeline for diffusers for training and evaluation.
    Usually includes following components:
    - vae
    - text_pipeline
        - text_encoder
        - processor
        - tokenizer
    - transformer (can be replaced by unet)
    - scheduler

    To enable FSDP, fsdp_configs should be provided (compulsory)
    and the method setup_fsdp_modules (optional).
    - fsdp_configs: a dict of following format

        module_name:               # the path attribute to the module required to be wrapped
            iterable: true/false   # whether the module is iterable
            **other_kwargs         # other kwargs that will be passed into
                                   # `torch.distributed.fsdp.fully_shard`

    - setup_fsdp_modules: a method to wrap modules iteratively
        By default, the BasePipline implements it with the above component names, override is
        required if there is a different component name.

    """

    vae: ModelMixin = dataclasses.field(init=None)
    text_pipeline: DiffusionPipeline = dataclasses.field(init=None)
    transformer: ModelMixin | PeftAdapterMixin = dataclasses.field(init=None)
    scheduler: SchedulerMixin = dataclasses.field(init=None)
    fsdp_configs: dict | None = None
    fsdp_modules: list | None = None

    @property
    def summary(self) -> dict[str, dict[str, int | float]]:
        return {
            "text_encoder": summarize_model(self.text_pipeline.text_encoder),
            "transformer": summarize_model(self.transformer),
            "vae": summarize_model(self.vae),
        }

    @property
    def fsdp_module_configs(self) -> list[dict[str, Any]]:
        if self.fsdp_configs is not None:
            module_configs = []
            for module_name, configs in self.fsdp_configs.items():
                module = get_nested_attr(self, module_name)
                is_iterable = configs.pop("iterable", False)
                if is_iterable:
                    module_configs.extend([{"module": m, "configs": configs} for m in module])
                else:
                    module_configs.append({"module": module, "configs": configs})
            return module_configs
        return []

    @property
    def trainable_params(self) -> list[torch.Tensor]:
        pass

    def setup_fsdp_modules(self, fsdp_strategy: FSDPStrategy, device: torch.device, dtype: torch.dtype):
        # VAE: Full parameters to all devices
        self.vae.to(device, dtype=dtype)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        mesh = parallel_handler.get_device_mesh(fsdp_strategy)

        if FSDPStrategy.is_no_shard(fsdp_strategy):
            if self.fsdp_configs is not None:
                logger.warning(f"FSDPStrategy is {fsdp_strategy}, fsdp_configs will be ignored.")

            self.text_pipeline.to(device, dtype=dtype)
            self.transformer.to(device, dtype=dtype)

            self.fsdp_modules = [self.transformer, self.text_pipeline.text_encoder]

        elif FSDPStrategy.is_full_shard(fsdp_strategy):
            assert self.fsdp_configs is not None, f"FSDPStrategy is {fsdp_strategy}, but fsdp_configs are not given."
            module_configs = []
            for module_name, raw_configs in self.fsdp_configs.items():
                configs: dict[str, Any] = copy.deepcopy(raw_configs)

                is_iterable = configs.pop("iterable", False)
                reshard_after_forward = configs.pop("reshard_after_forward", True)

                module = get_nested_attr(self, module_name)

                if is_iterable:
                    for submodule in module:
                        module_configs.append((None, submodule, configs))
                else:
                    module_configs.append((module_name, module, configs))

            for module_name, module, configs in module_configs:
                fsdp_kwargs = dict(
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                    **configs,
                )
                wrapped = fully_shard(module, **fsdp_kwargs)
                if module_name is not None:
                    set_nested_attr(self, module_name, wrapped)

            self.fsdp_modules = [self.transformer, self.text_pipeline.text_encoder]
        else:
            logger.warning(f"Unsupported FSDPStrategy: {fsdp_strategy}.")
        return self

    def forward_step(self, batch):
        pass

    @torch.inference_mode()
    def eval_step(self, batch, global_step, num_inference_steps: int = 50, cfg_scale: float = 0.0):
        pass

    @torch.inference_mode()
    def generate(self, prompt: str, image: Image.Image | list[Image.Image] | None = None, **kwargs):
        pass
