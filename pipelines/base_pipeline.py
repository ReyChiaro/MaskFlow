import torch
import dataclasses

from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from torch.distributed.fsdp import fully_shard

from functools import reduce
from PIL import Image
from typing import Any

from utils.summary import summarize_model


@dataclasses.dataclass
class BasePipeline:

    vae: ModelMixin = dataclasses.field(init=None)
    text_pipeline: DiffusionPipeline = dataclasses.field(init=None)
    transformer: ModelMixin | PeftAdapterMixin = dataclasses.field(init=None)
    scheduler: SchedulerMixin = dataclasses.field(init=None)
    fsdp_configs: dict | None = None

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
                module = reduce(getattr, module_name.split("."), self)
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

    def forward_step(self, batch):
        pass

    @torch.inference_mode()
    def eval_step(self, batch, global_step, num_inference_steps: int = 50, cfg_scale: float = 0.0):
        pass

    @torch.inference_mode()
    def generate(self, prompt: str, image: Image.Image | list[Image.Image] | None = None, **kwargs):
        pass
