import torch
import dataclasses

from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from loguru import logger
from PIL import Image
from typing import Any, Iterable

from utils.summary import summarize_model
from pipelines.utils import get_nested_attr
from trainer.parallel.handler import parallel_handler
from trainer.parallel.fsdp_strategy import FSDPStrategy


@dataclasses.dataclass
class PreprocessOutput:

    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    vlm_conditions: list[torch.Tensor] | dict[str, torch.Tensor] | None = None
    dit_conditions: list[torch.Tensor] | dict[str, torch.Tensor] | None = None
    target: torch.Tensor | None = None


@dataclasses.dataclass
class ForwardOutput:

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor

    height: int
    width: int

    noise: torch.Tensor | None = None
    noised_target: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    ground_truth: torch.Tensor | None = None
    conditions: list[torch.Tensor] | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds_mask: torch.Tensor | None = None


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

    _fsdp_module_configs: list[dict] | None = None

    @property
    def summary(self) -> dict[str, dict[str, int | float]]:
        return {
            "text_encoder": summarize_model(self.text_pipeline.text_encoder),
            "transformer": summarize_model(self.transformer),
            "vae": summarize_model(self.vae),
        }

    @property
    def fsdp_module_configs(self) -> list[dict[str, Any]]:
        if self._fsdp_module_configs is None:
            self._fsdp_module_configs = []
            if self.fsdp_configs is not None:
                module_configs = []
                for module_name, raw_configs in self.fsdp_configs.items():
                    module = get_nested_attr(self, module_name)
                    configs = dict(raw_configs)
                    is_iterable = configs.pop("iterable", isinstance(module, Iterable))
                    if is_iterable:
                        for i, m in enumerate(module):
                            module_configs.append({"name": f"{module_name}.{i}", "module": m, "configs": configs})
                    else:
                        module_configs.append({"name": module_name, "module": module, "configs": configs})
                self._fsdp_module_configs = module_configs
        return self._fsdp_module_configs

    @property
    def trainable_params(self) -> list[torch.Tensor]:
        r"""
        Return the trainable params list of tensors.
        """
        params = []
        for m in [self.vae, getattr(self.text_pipeline, "text_encoder", None), self.transformer]:
            if m is None:
                continue
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)
        return params

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
            # Whole modules will be loaded into each device.
            if self.fsdp_configs is not None:
                logger.warning(f"FSDPStrategy is {fsdp_strategy}, fsdp_configs will be ignored.")

            self.text_pipeline.to(device, dtype=dtype)
            self.transformer.to(device, dtype=dtype)

            self.fsdp_modules = [self.transformer, self.text_pipeline.text_encoder]

        elif FSDPStrategy.is_full_shard(fsdp_strategy):
            assert self.fsdp_configs is not None, f"FSDPStrategy is {fsdp_strategy}, but fsdp_configs are not given."
            # module_configs = []
            # for module_name, raw_configs in self._fsdp_module_configs:
            #     configs: dict[str, Any] = copy.deepcopy(raw_configs)

            #     # is_iterable = configs.pop("iterable", False)
            #     reshard_after_forward = configs.pop("reshard_after_forward", True)

            #     module = get_nested_attr(self, module_name)

            #     if is_iterable:
            #         for submodule in module:
            #             module_configs.append((None, submodule, configs))
            #     else:
            #         module_configs.append((module_name, module, configs))

            for module_configs in self.fsdp_module_configs:
                # module_name = module_configs["module_name"]
                module = module_configs["module"]
                configs = module_configs["configs"]
                fsdp_kwargs = dict(mesh=mesh, mp_policy=mp_policy, **configs)
                # In-place
                fully_shard(module, **fsdp_kwargs)
                # if module_name is not None:
                #     set_nested_attr(self, module_name, wrapped)

            self.fsdp_modules = [self.transformer, self.text_pipeline.text_encoder]

        else:
            logger.warning(f"Unsupported FSDPStrategy: {fsdp_strategy}.")
        return self

    def setup_additional_transformer(
        self,
        transformer: torch.nn.Module,
        fsdp_strategy: FSDPStrategy,
        device: torch.device,
        dtype: torch.dtype,
    ):
        r"""Apply the pipeline's transformer placement policy to another transformer."""
        if FSDPStrategy.is_no_shard(fsdp_strategy):
            transformer.to(device, dtype=dtype)
            return transformer

        if not FSDPStrategy.is_full_shard(fsdp_strategy):
            logger.warning(f"Unsupported FSDPStrategy: {fsdp_strategy}.")
            return transformer

        mp_policy = MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        mesh = parallel_handler.get_device_mesh(fsdp_strategy)
        for module_name, raw_configs in self.fsdp_configs.items():
            if module_name != "transformer" and not module_name.startswith("transformer."):
                continue

            configs = dict(raw_configs)
            is_iterable = configs.pop("iterable", False)
            relative_name = module_name.removeprefix("transformer.")
            module = transformer if module_name == "transformer" else get_nested_attr(transformer, relative_name)
            modules = module if is_iterable else [module]
            for submodule in modules:
                fully_shard(submodule, mesh=mesh, mp_policy=mp_policy, **configs)
        return transformer

    def forward_step(self, batch, **kwargs):
        pass

    @torch.inference_mode()
    def eval_step(self, batch, num_inference_steps: int, **kwargs):
        pass

    @torch.inference_mode()
    def generate(self, prompt: str, image: Image.Image | list[Image.Image] | None = None, **kwargs):
        pass
