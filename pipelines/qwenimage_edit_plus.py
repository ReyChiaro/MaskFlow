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

from torchvision.utils import save_image
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Literal, Any

from schedulers.flow_matching import RectifiedFlowMatchingScheduler
from pipelines.pipeline_manager import DiTPipelineManager, DenoiserInputs


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
            torch_dtype=torch_dtype,
        ).to(device)

        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        ).to(device)

        self.text_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            pretrained_model_name_or_path,
            scheduler=None,
            vae=None,
            transformer=None,
            torch_dtype=torch_dtype,
        ).to(device)

        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.text_pipeline.text_encoder.requires_grad_(False)

    def preprocess_everything(self, batch, device, dtype):
        r"""
        Preprocess batched data, conduct reshape or any other augmentations on it.
        """
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
        batch["conditions_to_text_pipeline"] = [c.to(device, dtype=dtype) for c in small_conditions]

        return batch

    def encode_prompt(
        self,
        prompt: str | list[str],
        conditions: torch.Tensor | list[torch.Tensor] | None = None,
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
        if not isinstance(conditions, list):
            conditions = [conditions]
        return self.text_pipeline.encode_prompt(
            prompt=prompt,
            image=conditions,
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
        vae_latents = retrieve_latents(self.vae.encode(image), generator, sample_mode).to(device, dtype)
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
        return (vae_latents - latents_mean) / latents_std

    def encode_everything(self, batch, generator, device, dtype):
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=batch.get("prompt", None),
            conditions=batch.get("conditions_to_text_pipeline", None),
            prompt_embeds=batch.get("prompt_embeds", None),
            prompt_embeds_mask=batch.get("prompt_embeds_mask", None),
            device=device,
        )

        input_shapes = []
        target = self.encode_image(image=batch["target"], generator=generator, device=device, dtype=dtype)
        B, C, _, H, W = target.shape
        input_shapes.append((1, H // self.transformer_pacth_size, W // self.transformer_pacth_size))
        target_latents = QwenImageEditPlusPipeline._pack_latents(target, B, C, H, W)

        conditions = [
            self.encode_image(image=img, generator=generator, sample_mode="sample", device=device, dtype=dtype)
            for img in batch.get("conditions", [])
        ]
        input_shapes.extend(
            [
                (1, c.shape[-2] // self.transformer_pacth_size, c.shape[-1] // self.transformer_pacth_size)
                for c in conditions
            ]
        )
        input_shapes = [input_shapes] * B
        condition_latents = [
            QwenImageEditPlusPipeline._pack_latents(
                latent, latent.shape[0], latent.shape[1], latent.shape[-2], latent.shape[-1]
            )
            for latent in conditions
        ]

        noise = torch.randn_like(target_latents, generator=generator, device=device, dtype=dtype)
        t = self.scheduler.sample_timesteps(B, generator=generator, device=device, dtype=dtype)
        mu = self.scheduler.calculate_shift_mu(target_latents.shape[1])
        sigmas = self.scheduler.time_shift(t, mu)

        return {
            "height": batch["target"].shape[-2],
            "width": batch["target"].shape[-1],
            "timesteps": sigmas,
            "mu": mu,
            "sigmas": sigmas,
            "noise": noise,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "target_latents": target_latents,
            "condition_latents": condition_latents,
            "image_shapes": input_shapes,
        }

    def add_noise(self, eps: torch.Tensor, x0: torch.Tensor, conditions: list[torch.Tensor], sigmas: torch.Tensor):
        noisy_latents = self.scheduler.add_noise(sigmas, eps, x0)
        truth_latents = self.scheduler.get_ground_truth(eps, x0)
        return noisy_latents, truth_latents

    def compute_loss(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        r"""Compute loss without masks and weights"""
        return (
            F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
            .reshape(predictions.shape[0], -1)
            .mean(dim=1)
        ).mean()

    def decode_image(self, predictions: torch.Tensor, device, dtype, output_type: str = "pt") -> torch.Tensor:
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
        predictions = predictions * latents_std + latents_mean
        predictions = self.vae.decode(predictions, return_dict=False)[0][:, :, 0]
        predictions = self.text_pipeline.image_processor.postprocess(predictions, output_type=output_type)
        return predictions

    def denoise(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list[list[tuple[int]]],
        img_seq_len: int,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        predictions: torch.Tensor = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            attention_kwargs=kwargs.get("attention_kwargs", {}),
            return_dict=False,
        )[0]
        predictions = predictions[:, :img_seq_len]
        return predictions.to(dtype=dtype)

    def forward_step(self, batch, generator=None, device=None, dtype=None):
        inputs = self.preprocess_everything(batch, device, dtype)
        inputs = self.encode_everything(inputs, generator, device, dtype)
        noisy_latents, truth_latents = self.add_noise(
            inputs["noise"], inputs["target_latents"], inputs["condition_latents"], inputs["sigmas"]
        )
        hidden_states = torch.cat([noisy_latents] + [c for c in inputs["condition_latents"]], dim=1)
        predictions = self.denoise(
            hidden_states,
            inputs["timesteps"],
            inputs["prompt_embeds"],
            inputs["prompt_embeds_mask"],
            inputs["image_shapes"],
            noisy_latents.shape[1],
        )
        loss = self.compute_loss(predictions, truth_latents)
        return loss

    @torch.inference_mode()
    def eval_step(self, batch, num_inference_steps: int, generator=None, device=None, dtype=None) -> torch.Tensor:
        r"""
        Evaluation when training, the ground truths are provided.
        """
        from tqdm import tqdm

        inputs = self.preprocess_everything(batch, device, dtype)
        inputs = self.encode_everything(inputs, generator, device, dtype)

        denoised = inputs["noise"]
        for delta_sigma, t in tqdm(
            self.scheduler.inference_delta_sigmas(num_inference_steps, inputs["mu"]), total=num_inference_steps
        ):
            hidden_states = torch.cat([denoised] + [c for c in inputs["condition_latents"]], dim=1)
            t = t.expand(hidden_states.shape[0]).to(device, dtype=dtype)
            delta_sigma = delta_sigma.to(device, dtype=dtype)
            predictions = self.denoise(
                hidden_states,
                t,
                inputs["prompt_embeds"],
                inputs["prompt_embeds_mask"],
                inputs["image_shapes"],
                denoised.shape[1],
            )
            denoised = denoised + delta_sigma * predictions

        denoised = QwenImageEditPlusPipeline._unpack_latents(
            denoised,
            height=inputs["height"],
            width=inputs["width"],
            vae_scale_factor=self.vae_scale_factor,
        )
        output = self.decode_image(denoised, device, dtype)
        return output
