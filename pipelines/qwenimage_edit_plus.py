import torch
import dataclasses

from diffusers import QwenImageEditPlusPipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import retrieve_latents

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from typing import Literal, Any
from loguru import logger

from schedulers.flow_matching import RectifiedFlowMatchingScheduler
from pipelines.pipeline_manager import DiTPipelineManager, DenoiserInputs


@dataclasses.dataclass
class QwenImageTransformerInputs(DenoiserInputs):

    img_shapes: list[list[tuple]] = dataclasses.field(default_factory=list())
    encoder_hidden_states: torch.Tensor | None = None
    encoder_hidden_states_mask: torch.Tensor | None = None


class QwenImageEditPlusManager(DiTPipelineManager):

    def __init__(self):
        super().__init__()

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** len(self.vae.config.temperal_downsample) if getattr(self, "vae", None) else 8

    @property
    def transformer_pacth_size(self) -> int:
        return self.transformer.config.patch_size if getattr(self, "transformer", None) else 2

    def init_components(self, pretrained_model_name_or_path, torch_dtype, device):
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        self.vae = AutoencoderKLQwenImage.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        ).to(device, dtype=torch_dtype)

        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
        ).to(device, dtype=torch_dtype)

        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
        ).to(device, dtype=torch_dtype)

        self.text_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            scheduler=None,
            vae=None,
            transformer=None,
        ).to(device, dtype=torch_dtype)

        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        for _, c in self.text_pipeline.components.items():
            if hasattr(c, "requires_grad_"):
                c.requires_grad_(False)

    def encode_prompt(
        self,
        prompt: str | list[str],
        image_conditions: torch.Tensor | list[torch.Tensor] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            prompt (str or list of str):
            image_conditions (Tensor or list of Tensor or None): [B, C, 1, H, W] Support multiple conditions for one prompt.
        """
        if self.text_pipeline is None or prompt_embeds is not None:
            return prompt_embeds, prompt_embeds_mask

        # NOTE: QwenImageEditPlus encode_prompt should pass list of tensors for multi-image forward
        if not isinstance(image_conditions, list):
            image_conditions = [image_conditions]
        return self.text_pipeline.encode_prompt(
            prompt=prompt,
            image=image_conditions,
            device=device,
            num_images_per_prompt=1,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )

    def encode_image(
        self,
        image: torch.Tensor,
        generator: torch.Generator,
        sample_mode: Literal["argmax", "sample", "latents"] = "argmax",
    ) -> torch.Tensor:
        r"""
        Args:
            image (Tensor or list of Tensor): [B, C, 1, H, W]

        Return:
            Tensor [B, z_dim, 1, H//vae_scale_factor, W//vae_scale_factor]
        """
        vae_latents = retrieve_latents(
            self.vae.enocde(image),
            generator=generator,
            sample_mode=sample_mode,
        )
        latents_mean = torch.full(
            (1, self.vae.config.z_dim, 1, 1, 1),
            self.vae.config.latents_mean,
            device=image.device,
            dtype=image.dtype,
        )
        latents_std = torch.full(
            (1, self.vae.config.z_dim, 1, 1, 1),
            self.vae.config.latents_std,
            device=image.device,
            dtype=image.dtype,
        )
        return (vae_latents - latents_mean) / latents_std

    def preprocess_denoiser_inputs(
        self,
        prompt_embeds: torch.Tensor,
        noisy_target_latents: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None = None,
        image_condition_latents: list[torch.Tensor] | torch.Tensor | None = None,
    ):
        r"""
        Args:
            prompt_embeds (Tensor): [B, L, C]
            noisy_target_latents (Tensor): [B, C, 1, H, W]
            image_condition_latents (Tensor or list of Tensor or None): [B, C, 1, H, W]
        """
        # ---------------- Process Images ---------------- #
        # Pack and reshape latents to [B, L, C]
        noisy_target_H, noisy_target_W = noisy_target_latents.shape[-2:]
        noisy_target_latents = QwenImageEditPlusPipeline._pack_latents(
            latents=noisy_target_latents,
            batch_size=noisy_target_latents.shape[0],
            num_channels_latents=noisy_target_latents.shape[1],
            height=noisy_target_latents.shape[3],
            width=noisy_target_latents.shape[4],
        )
        input_latents = torch.cat([prompt_embeds, noisy_target_latents], dim=1)
        input_shapes = [
            (1, noisy_target_H // self.transformer_pacth_size, noisy_target_W // self.transformer_pacth_size)
        ]

        if image_condition_latents is not None:
            if not isinstance(image_condition_latents, list):
                image_condition_latents = [image_condition_latents]
            for image_conds in image_condition_latents:
                cond_H, cond_W = image_conds.shape[-2:]
                image_conds = QwenImageEditPlusPipeline._pack_latents(
                    latents=image_conds,
                    batch_size=image_conds.shape[0],
                    num_channels_latents=image_conds.shape[1],
                    height=image_conds.shape[3],
                    width=image_conds.shape[4],
                )
                input_shapes.append((1, cond_H // self.transformer_pacth_size, cond_W // self.transformer_pacth_size))
                input_latents = torch.cat([input_latents, image_condition_latents], dim=1)

        input_shapes = [input_shapes] * noisy_target_latents.shape[0]

        return QwenImageTransformerInputs(
            hidden_states=input_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=input_shapes,
        )

    def postprocess_denoiser_outputs(self):
        pass

    def denoise(
        self,
        denoiser_inputs: QwenImageTransformerInputs,
        timestep: float,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            timestep (float): Range [0, 1]
        """
        predictions = self.transformer(
            hidden_states=denoiser_inputs.hidden_states,
            timestep=timestep,
            encoder_hidden_states_mask=denoiser_inputs.encoder_hidden_states_mask,
            encoder_hidden_states=denoiser_inputs.encoder_hidden_states,
            img_shapes=denoiser_inputs.img_shapes,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        return predictions
