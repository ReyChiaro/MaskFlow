import torch
import dataclasses

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from typing import Literal, Any
from PIL import Image
from loguru import logger

from utils.summary import summarize_model


@dataclasses.dataclass
class DenoiserInputs:

    hidden_states: torch.Tensor


class DiTPipelineManager:

    def __init__(self, pretrained_model_name_or_path: str):
        self.text_encoder = None
        self.text_pipeline = None
        self.vae = None
        self.transformer = None
        self.scheduler = None
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    @property
    def trainable_parameters(self) -> dict[str, list[torch.Tensor]]:
        r"""
        Summary the trainable parameters of the trainable modules,
        if there is any trainable adapter, it will be considerred.
        """
        param_dict = {}
        param_dict["transformer"] = [p for p in self.transformer.parameters() if p.requires_grad]
        param_dict["vae"] = [p for p in self.transformer.parameters() if p.requires_grad]
        param_dict["text_encoder"] = [p for p in self.transformer.parameters() if p.requires_grad]
        return param_dict

    @property
    def trainable_modules(self) -> list[str]:
        return [k for k, v in self.pipeline_manager.trainable_parameters.items() if v]

    @property
    def summary(self) -> dict[str, dict[str, int | float]]:
        return {
            "text_encoder": summarize_model(self.text_encoder) if self.text_encoder is not None else {},
            "transformer": summarize_model(self.transformer),
            "vae": summarize_model(self.vae),
        }

    def init_components(
        self,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype,
        device: torch.device | str,
    ):
        pass

    def encode_prompt(
        self,
        prompt: str | list[str],
        image_conditions: torch.Tensor | list[torch.Tensor] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def encode_image(
        self,
        image: torch.Tensor,
        generator: torch.Generator,
        sample_mode: Literal["argmax", "sample", "latents"] = "argmax",
    ) -> torch.Tensor:
        pass

    def preprocess_denoiser_inputs(
        self,
        prompt_embeds: torch.Tensor,
        noisy_target_latents: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None = None,
        image_condition_latents: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> DenoiserInputs:
        pass

    def denoise(
        self,
        denoiser_inputs: DenoiserInputs,
        timestep: float,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        pass

    def postprocess_denoiser_outputs(self):
        pass
