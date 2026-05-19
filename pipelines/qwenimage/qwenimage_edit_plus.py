import copy
import torch
import torch.nn.functional as F

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
    CONDITION_IMAGE_SIZE,
    calculate_dimensions,
)
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from dataclasses import dataclass, field
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Any, Literal

from schedulers import RectifiedFlowMatchingScheduler
from pipelines import BasePipeline


@dataclass
class QwenImageEditPlus(BasePipeline):

    pretrained_model: str
    scheduler: RectifiedFlowMatchingScheduler
    generator: torch.Generator
    device: torch.device
    dtype: torch.dtype

    vae: AutoencoderKLQwenImage = field(init=False, default=None)
    transformer: QwenImageTransformer2DModel = field(init=False, default=None)
    text_pipeline: QwenImageEditPlusPipeline = field(init=False, default=None)

    def __post_init__(self):
        self.vae = (
            AutoencoderKLQwenImage.from_pretrained(self.pretrained_model, subfolder="vae", torch_dtype=self.dtype)
            .to(self.device)
            .requires_grad_(False)
        )
        self.transformer = (
            QwenImageTransformer2DModel.from_pretrained(
                self.pretrained_model, subfolder="transformer", torch_dtype=self.dtype
            )
            .to(self.device)
            .requires_grad_(False)
        )
        self.text_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.pretrained_model, vae=None, transformer=None, torch_dtype=self.dtype
        ).to(self.device)
        self.text_pipeline.text_encoder.requires_grad_(False)
        self.image_processor = self.text_pipeline.image_processor

    @property
    def trainable_params(self) -> list[torch.Tensor]:
        return [p for p in self.transformer.parameters() if p.requires_grad]

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** len(self.vae.config.temperal_downsample) if getattr(self, "vae", None) else 8

    @property
    def vae_channels(self) -> int:
        return self.vae.config.z_dim

    @property
    def pacth_size(self) -> int:
        return self.transformer.config.patch_size if getattr(self, "transformer", None) else 2

    def preprocess_inputs(self, batch) -> dict[str, Any]:
        r"""
        A batched data is supposed to have keys `prompt`, `conditions` and `target`.
        """
        raw_batch = copy.deepcopy(batch)
        prompt: str | list[str] = batch["prompt"]
        conditions: list[torch.Tensor] = batch["conditions"]
        target: torch.Tensor = batch["target"]

        # ---------------- Preprocess ---------------- #
        # To tensor and reshape to target areas
        h, w = target.shape[-2:]
        aspect = w / h
        cond_h, cond_w = calculate_dimensions(CONDITION_IMAGE_SIZE, aspect)
        conditions_vlm = [self.image_processor.resize(c, cond_h, cond_w) for c in conditions]
        target = self.image_processor.preprocess(target, h, w).unsqueeze(2)
        conditions_dit = [self.image_processor.preprocess(c, h, w).unsqueeze(2) for c in conditions]

        return {
            "raw": raw_batch,
            "prompt": prompt,
            "conditions_vlm": [c.to(self.device, dtype=self.dtype) for c in conditions_vlm],
            "conditions_dit": [c.to(self.device, dtype=self.dtype) for c in conditions_dit],
            "target": target.to(self.device, dtype=self.dtype),
        }

    def encode_prompt(self, prompt: str, conditions: torch.Tensor | list[torch.Tensor]) -> tuple[torch.Tensor]:
        if not isinstance(conditions, (tuple, list)):
            conditions = [conditions]
        return self.text_pipeline.encode_prompt(prompt=prompt, image=conditions, device=self.device)

    def encode_image(
        self, image: torch.Tensor, sample_mode: Literal["argmax", "sample", "latents"] = "sample"
    ) -> torch.Tensor:
        vae_outputs = self.vae.encode(image)
        if sample_mode == "sample":
            latents = vae_outputs.latent_dist.sample(self.generator)
        elif sample_mode == "argmax":
            latents = vae_outputs.latent_dist.mode()
        else:
            # sample mode is `latents` or any other
            latents = vae_outputs.latents

        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents = (latents - latents_mean) / latents_std
        return latents

    def decode_image(self, xt: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=xt.device,
            dtype=xt.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=xt.device,
            dtype=xt.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        xt = xt * latents_std + latents_mean
        xt = self.vae.decode(xt, return_dict=False)[0][:, :, 0]
        xt = self.text_pipeline.image_processor.postprocess(xt, output_type="pt")
        return xt

    def prepare_forward_inputs(self, batch):
        r"""
        Prepare training forward inputs.
        The sample mode for VAE is fixed to `sample`, `target` must be provided.
        """
        sample_mode = "sample"
        processed_data = self.preprocess_inputs(batch)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            processed_data["prompt"], processed_data["conditions_vlm"]
        )

        image_shapes = []
        conditions_dit = processed_data["conditions_dit"]
        target = processed_data["target"]

        # ---------------- Encode and Pack ---------------- #
        # Encode
        tgt = self.encode_image(target, sample_mode)
        conds = [self.encode_image(c, sample_mode) for c in conditions_dit]
        image_shapes.append((1, tgt.shape[-2] // self.pacth_size, tgt.shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * tgt.shape[0]

        # Pack to 3D
        x0 = QwenImageEditPlusPipeline._pack_latents(tgt, tgt.shape[0], tgt.shape[1], tgt.shape[-2], tgt.shape[-1])
        cond_latents = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]

        # --------------- Sample and Add Noise -------------- #
        noise = torch.randn_like(x0, generator=self.generator)
        ts = self.scheduler.sample_timesteps(x0.shape[0], self.generator, self.device)
        xt, sigmas = self.scheduler.add_noise(noise, x0, ts)
        gt = self.scheduler.get_velocity(noise, x0)

        return {
            "height": target.shape[-2],
            "width": target.shape[-1],
            "timesteps": ts,
            "sigmas": sigmas,
            "noise": noise,
            "gt": gt,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "xt": xt,
            "x0": x0,
            "conditions": cond_latents,
            "image_shapes": image_shapes,
        }

    def prepare_eval_inputs(self, batch):
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`, `target` must be provided.
        """
        sample_mode = "argmax"
        processed_data = self.preprocess_inputs(batch)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            processed_data["prompt"], processed_data["conditions_vlm"]
        )

        image_shapes = []
        conditions_dit = processed_data["conditions_dit"]
        target = processed_data["target"]

        # ---------------- Encode and Pack ---------------- #
        # Encode
        noise_shape = (
            target.shape[0],
            self.vae_channels,
            1,
            target.shape[-2] // self.vae_scale_factor,
            target.shape[-1] // self.vae_scale_factor,
        )
        noise = torch.randn(noise_shape, generator=self.generator, device=self.device, dtype=self.dtype)
        conds = [self.encode_image(c, sample_mode) for c in conditions_dit]
        image_shapes.append((1, noise_shape[-2] // self.pacth_size, noise_shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * noise_shape[0]

        # Pack to 3D
        noise = QwenImageEditPlusPipeline._pack_latents(
            noise, noise.shape[0], noise.shape[1], noise.shape[-2], noise.shape[-1]
        )
        cond_latents = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]

        return {
            "height": target.shape[-2],
            "width": target.shape[-1],
            "noise": noise,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "conditions": cond_latents,
            "image_shapes": image_shapes,
        }

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

    def compute_loss(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        r"""Compute loss without masks and weights"""
        return (
            F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
            .reshape(predictions.shape[0], -1)
            .mean(dim=1)
        ).mean()

    def forward_step(self, batch):
        inputs = self.prepare_forward_inputs(batch)
        hidden_states = torch.cat([inputs["xt"]] + [c for c in inputs["conditions"]], dim=1)
        predictions = self.denoise(
            hidden_states,
            inputs["sigmas"],
            inputs["prompt_embeds"],
            inputs["prompt_embeds_mask"],
            inputs["image_shapes"],
            inputs["xt"].shape[1],
        )
        loss = self.compute_loss(predictions, inputs["gt"])
        return loss

    @torch.inference_mode()
    def eval_step(self, batch, num_inference_steps: int = 50, cfg: float = 0.0):
        r"""
        TODO: Support CFG
        """
        from tqdm import tqdm
        inputs = self.prepare_eval_inputs(batch)

        xt = inputs["noise"]
        with self.scheduler.inference_sampler(xt, num_inference_steps, xt.shape[1]) as sampler:
            for xt, t, inferencer in tqdm(sampler, total=num_inference_steps):
                hidden_states = torch.cat([xt] + [c for c in inputs["conditions"]], dim=1)
                pred = self.denoise(
                    hidden_states,
                    t.expand(hidden_states.shape[0]).to(device=self.device, dtype=self.dtype),
                    inputs["prompt_embeds"],
                    inputs["prompt_embeds_mask"],
                    inputs["image_shapes"],
                    xt.shape[1],
                )
                inferencer.step(pred)
        output = QwenImageEditPlusPipeline._unpack_latents(xt, inputs["height"], inputs["width"], self.vae_scale_factor)
        output = self.decode_image(output)
        return output

    @torch.inference_mode()
    def generate(self, prompt, images, **kwargs):
        return super().generate(prompt, images, **kwargs)
