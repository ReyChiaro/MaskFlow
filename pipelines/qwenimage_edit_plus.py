import math
import torch
import dataclasses
import torch.nn.functional as F

from diffusers import QwenImageEditPlusPipeline, AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    retrieve_latents,
    calculate_dimensions,
    CONDITION_IMAGE_SIZE,
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Literal, Any
from loguru import logger

from schedulers.flow_matching import RectifiedFlowMatchingScheduler
from pipelines.pipeline_manager import DiTPipelineManager, DenoiserInputs


@dataclasses.dataclass
class QwenImageTransformerInputs(DenoiserInputs):

    target_height: int
    target_width: int
    img_shapes: list[list[tuple]] = dataclasses.field(default_factory=list())
    encoder_hidden_states: torch.Tensor | None = None
    encoder_hidden_states_mask: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    attention_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict())
    ground_truths: torch.Tensor | None = None


@dataclasses.dataclass
class QwenImageTransformerOutputs:

    target_height: int
    target_width: int
    predictions: torch.Tensor
    timesteps: torch.Tensor | None = None
    ground_truths: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class QwenImageEditPlusManager(DiTPipelineManager):

    def __init__(self, pretrained_model_name_or_path: str, scheduler_configs: OmegaConf | None = None):
        super().__init__(pretrained_model_name_or_path)
        self.scheduler_configs = scheduler_configs

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** len(self.vae.config.temperal_downsample) if getattr(self, "vae", None) else 8

    @property
    def transformer_pacth_size(self) -> int:
        return self.transformer.config.patch_size if getattr(self, "transformer", None) else 2

    def init_components(self, pretrained_model_name_or_path, torch_dtype, device):
        if self.scheduler_configs is None:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="scheduler",
            )
        else:
            self.scheduler: RectifiedFlowMatchingScheduler = instantiate(self.scheduler_configs)

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

    def preprocess_everything(self, batch, device, dtype):
        prompt: str | list[str] = batch["prompt"]
        conditions: list[torch.Tensor] = batch["conditions"]
        target: torch.Tensor = batch["target"]

        # ---------------- Preprocess ---------------- #
        # To tensor and reshape to target areas
        _, _, H, W = target.shape
        tgt_aspect = W / H
        cond_H, cond_W = calculate_dimensions(CONDITION_IMAGE_SIZE, tgt_aspect)
        small_conditions = [self.text_pipeline.image_processor.resize(c, cond_H, cond_W) for c in conditions]
        target = self.text_pipeline.image_processor.preprocess(target, H, W).unsqueeze(2)
        conditions = [self.text_pipeline.image_processor.preprocess(c, H, W).unsqueeze(2) for c in conditions]

        batch["prompt"] = prompt
        batch["conditions"] = [c.to(device, dtype=dtype) for c in conditions]
        batch["target"] = target.to(device, dtype=dtype)
        batch["conditions_to_text_encoder"] = [c.to(device, dtype=dtype) for c in small_conditions]

        print(f"{prompt=}")
        print(f"{batch['target'].shape=} | {batch['target'].device=} | {batch['target'].dtype=}")
        print(
            f"conditions shapes={[c.shape for c in conditions]} | conditions devices={[c.device for c in conditions]}"
        )
        print(f"conditions to text encoder shapes={[c.shape for c in small_conditions]}")
        return batch

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
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            image (Tensor or list of Tensor): [B, C, 1, H, W]

        Return:
            Tensor [B, z_dim, 1, H//vae_scale_factor, W//vae_scale_factor]
        """
        vae_latents = retrieve_latents(
            self.vae.encode(image),
            generator=generator,
            sample_mode=sample_mode,
        ).to(device, dtype)
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=device,
            dtype=dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=device,
            dtype=dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        print(device, dtype)
        print(vae_latents.device, vae_latents.dtype)
        print(latents_mean.device, latents_mean.dtype)
        print(latents_std.device,latents_std.dtype)
        return (vae_latents - latents_mean) / latents_std

    def add_noise(self, target_latents: torch.Tensor, generator, device, dtype):
        r"""
        Add noises to target and return the ground truth,
        re-write to support other operations.
        """
        B = target_latents.shape[0]
        noise_latents = torch.randn_like(target_latents, device=device, dtype=dtype, generator=generator)
        img_seq_len = target_latents.shape[1]

        t = self.scheduler.sample_timesteps(B, generator, device, dtype)
        mu = self.scheduler.calculate_shift_mu(img_seq_len)
        sigmas = self.scheduler.time_shift(t, mu)
        print(f"{t.dtype=}")
        print(f"{sigmas.dtype=}")
        
        print(f"{noise_latents.shape=}")
        print(f"{target_latents.shape=}")
        noisy_target_latents = self.scheduler.add_noise(sigmas, noise_latents, target_latents)
        ground_truths = self.scheduler.get_ground_truth(noise_latents, target_latents)
        print(f"{noisy_target_latents.dtype=}")
        print(f"{ground_truths.dtype=}")

        return t, noisy_target_latents, ground_truths

    def preprocess_denoiser_inputs(
        self,
        prompt_embeds: torch.Tensor,
        target_latents: torch.Tensor,
        prompt_embeds_mask: torch.Tensor | None = None,
        image_condition_latents: list[torch.Tensor] | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Args:
            prompt_embeds (Tensor): [B, L, C]
            target_latents (Tensor): [B, C, 1, H, W]
            image_condition_latents (Tensor or list of Tensor or None): [B, C, 1, H, W]
        """
        # ---------------- Process Images ---------------- #
        # Pack and reshape latents to [B, L, C]
        B, C, _, target_H, target_W = target_latents.shape
        target_latents = QwenImageEditPlusPipeline._pack_latents(
            latents=target_latents,
            batch_size=B,
            num_channels_latents=C,
            height=target_H,
            width=target_W,
        )

        # ------------------ Add Noises ------------------ #
        t, noisy_target_latents, ground_truths = self.add_noise(
            target_latents=target_latents,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # --------------- Construct Inputs --------------- #
        input_latents = torch.cat([noisy_target_latents], dim=1)
        input_shapes = [(1, target_H // self.transformer_pacth_size, target_W // self.transformer_pacth_size)]

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
                input_latents = torch.cat([input_latents, image_conds], dim=1)

        input_shapes = [input_shapes] * noisy_target_latents.shape[0]

        return QwenImageTransformerInputs(
            target_height=target_H,
            target_width=target_W,
            hidden_states=input_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=input_shapes,
            timesteps=t,
            attention_kwargs={},
            ground_truths=ground_truths,
        )

    def compute_loss(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        r"""Compute loss without masks and weights"""
        return (
            F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
            .reshape(predictions.shape[0], -1)
            .mean(dim=-1)
        ).mean()

    def postprocess_denoiser_outputs[T](self, denoiser_outputs: QwenImageTransformerOutputs) -> T:
        predictions = QwenImageEditPlusPipeline._unpack_latents(
            predictions,
            height=denoiser_outputs.target_height * self.vae_scale_factor,
            width=denoiser_outputs.target_width * self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        return predictions

    def denoise(self, denoiser_inputs: QwenImageTransformerInputs) -> torch.Tensor:
        r"""
        Args:
            timestep (float or Tensor): Range [0, 1]
        """
        print(denoiser_inputs.encoder_hidden_states.dtype, denoiser_inputs.encoder_hidden_states.shape)
        print(denoiser_inputs.ground_truths.dtype, denoiser_inputs.ground_truths.shape)
        print(denoiser_inputs.hidden_states.dtype, denoiser_inputs.hidden_states.shape)
        predictions: torch.Tensor = self.transformer(
            hidden_states=denoiser_inputs.hidden_states,
            timestep=denoiser_inputs.timesteps,
            encoder_hidden_states_mask=denoiser_inputs.encoder_hidden_states_mask,
            encoder_hidden_states=denoiser_inputs.encoder_hidden_states,
            img_shapes=denoiser_inputs.img_shapes,
            attention_kwargs=denoiser_inputs.attention_kwargs,
            return_dict=False,
        )[0]
        # First batch, first element
        pred_length = (denoiser_inputs.target_height // 2) * (denoiser_inputs.target_width // 2)
        predictions = predictions[:, :pred_length]
        loss = self.compute_loss(predictions, denoiser_inputs.ground_truths)
        return QwenImageTransformerOutputs(
            target_height=denoiser_inputs.target_height,
            target_width=denoiser_inputs.target_width,
            predictions=predictions,
            timesteps=denoiser_inputs.timesteps,
            ground_truths=denoiser_inputs.ground_truths,
            loss=loss,
        )
