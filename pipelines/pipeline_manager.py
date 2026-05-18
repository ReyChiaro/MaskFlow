import torch
import dataclasses

from typing import Literal, Any

from schedulers.flow_matching import RectifiedFlowMatchingScheduler
from utils.summary import summarize_model


@dataclasses.dataclass
class DenoiserInputs:

    hidden_states: torch.Tensor


@dataclasses.dataclass
class DenoiserOutputs:

    predictions: torch.Tensor
    loss: torch.Tensor | None = None


class DiTPipelineManager:
    components = ["transformer", "vae", "text_pipeline", "scheduler"]

    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        self.text_pipeline = None
        self.vae = None
        self.transformer = None
        self.scheduler: RectifiedFlowMatchingScheduler = None
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    @property
    def trainable_parameters(self) -> list[torch.Tensor]:
        r"""
        Summary the trainable parameters of the trainable modules,
        if there is any trainable adapter, it will be considerred.
        Note that this property will traverse all modules whenever it be called.
        """
        return [p for p in self.transformer.parameters() if p.requires_grad]

    @property
    def summary(self) -> dict[str, dict[str, int | float]]:
        return {
            "text_encoder": summarize_model(self.text_pipeline.text_encoder) if self.text_pipeline is not None else {},
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

    def forward_step(
        self,
        batch: tuple[Any],
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            batch: Batched data

        Return:
            loss (Tensor)
        """
        pass

    @torch.inference_mode()
    def eval_step(
        self,
        batch: tuple[Any],
        num_inference_steps: int,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        pass

    def preprocess_everything[T](self, batch: T, device, dtype) -> T:
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
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ) -> torch.Tensor:
        pass

    def compute_loss(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        pass

    def denoise(self, denoiser_inputs: DenoiserInputs) -> torch.Tensor:
        pass
