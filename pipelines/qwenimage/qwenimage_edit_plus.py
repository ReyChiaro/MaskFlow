from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
)
from tqdm import tqdm

from data_module.utils import image_dimensions, resize_image
from diffusers.utils.torch_utils import randn_tensor
from pipelines.base_pipeline import BasePipeline, ForwardOutput, PreprocessOutput
from schedulers import RectifiedFlowMatchingScheduler


def resize_rgb(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize CPU float32 RGB [B, C, H, W] in [0, 1] using PIL Lanczos."""
    return torch.stack([resize_image(item, height, width) for item in image])


@dataclass
class QwenForwardOutput(ForwardOutput):
    image_shapes: list = field(default_factory=list)


@dataclass
class QwenImageEditPlus(BasePipeline):
    pretrained_model: str | None = None
    scheduler: RectifiedFlowMatchingScheduler | None = None
    generator: torch.Generator | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    max_condition_resolution: int = 384 * 384
    divisible_by: int = 32

    vae: AutoencoderKLQwenImage = field(init=False, default=None)
    transformer: QwenImageTransformer2DModel = field(init=False, default=None)
    text_pipeline: QwenImageEditPlusPipeline = field(init=False, default=None)

    def __post_init__(self):
        self.vae = (
            AutoencoderKLQwenImage.from_pretrained(self.pretrained_model, subfolder="vae", torch_dtype=self.dtype)
            .to(self.device)
            .requires_grad_(False)
        )
        self.transformer = self.load_transformer()
        self.text_pipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.pretrained_model, vae=None, transformer=None, torch_dtype=self.dtype
        ).to(self.device)
        self.text_pipeline.text_encoder.requires_grad_(False)
        self.image_processor = self.text_pipeline.image_processor

    def load_transformer(self) -> QwenImageTransformer2DModel:
        return (
            QwenImageTransformer2DModel.from_pretrained(
                self.pretrained_model,
                subfolder="transformer",
                torch_dtype=self.dtype,
            )
            .to(self.device)
            .requires_grad_(False)
        )

    @property
    def vae_scale_factor(self) -> int:
        return 2 ** len(self.vae.config.temperal_downsample) if getattr(self, "vae", None) else 8

    @property
    def vae_channels(self) -> int:
        return self.vae.config.z_dim

    @property
    def pacth_size(self) -> int:
        return self.transformer.config.patch_size if getattr(self, "transformer", None) else 2

    def preprocess_inputs(self, batch: dict[str, Any]) -> PreprocessOutput:
        r"""
        A batched data is supposed to have keys `prompt`(list[str]), `target`(Tensor).
        Optional batched data keys: `negative_prompt`(list[str]), `conditions`(dict[str,Tensor])
        """
        prompt: list[str] = batch["prompt"]
        target = batch.get("target")
        negative_prompt: list[str] | None = batch.get("negative_prompt")
        conditions: dict[str, torch.Tensor] = batch["conditions"]
        # Dataset and single-image entry point supply aligned CPU float32 [B, C, H, W] images.
        h, w = conditions["source"].shape[-2:]
        if target is not None:
            target = self.image_processor.preprocess(target.to(self.device, dtype=self.dtype), h, w).unsqueeze(2)

        vlm_conditions = {}
        dit_conditions = {}
        for key, image in conditions.items():
            ih, iw = image.shape[-2:]
            ch, cw = image_dimensions(iw / ih, self.max_condition_resolution, self.divisible_by)
            vlm_conditions[key] = resize_rgb(image, ch, cw).to(self.device, dtype=self.dtype)
            dit_conditions[key] = self.image_processor.preprocess(
                image.to(self.device, dtype=self.dtype), ih, iw
            ).unsqueeze(2)

        return PreprocessOutput(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vlm_conditions=vlm_conditions,
            dit_conditions=dit_conditions,
            target=target,
            height=h,
            width=w,
        )

    def encode_prompt(
        self,
        prompt: list[str],
        vlm_conditions: torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor]:
        if isinstance(vlm_conditions, dict):
            vlm_conditions = [c for c in vlm_conditions.values()]
        if not isinstance(vlm_conditions, (tuple, list)):
            vlm_conditions = [vlm_conditions]
        return self.text_pipeline.encode_prompt(prompt=prompt, image=vlm_conditions, device=self.device)

    def encode_image(
        self,
        image: torch.Tensor,
        sample_mode: Literal["argmax", "sample", "latents"] = "sample",
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

    def prepare_forward_inputs(self, preprocessed_data: PreprocessOutput) -> ForwardOutput:
        r"""
        Prepare training forward inputs.
        The sample mode for VAE is fixed to `sample`, `target` must be provided.
        """
        if preprocessed_data.target is None:
            raise ValueError("Training requires target images.")
        sample_mode = "sample"

        # Conduct CFG dropout
        prompt = preprocessed_data.prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, preprocessed_data.vlm_conditions)

        image_shapes = []
        dit_conditsions = preprocessed_data.dit_conditions
        target = preprocessed_data.target
        height, width = target.shape[-2:]

        # ---------------- Encode and Pack ---------------- #
        # Encode
        tgt = self.encode_image(target, sample_mode)
        conds = [self.encode_image(c, sample_mode) for c in dit_conditsions.values()]
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
        sigmas = self.scheduler.get_sigmas(ts, img_seq_len=x0.shape[1])
        xt = self.scheduler.add_noise_by_sigmas(noise, x0, sigmas)
        gt = self.scheduler.get_velocity(noise, x0)

        return QwenForwardOutput(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            noised_target=xt,
            conditions=cond_latents,
            ground_truth=gt,
            noise=noise,
            timesteps=sigmas,
            sigmas=sigmas,
            height=height,
            width=width,
            image_shapes=image_shapes,
        )

    def sample_eval_noise(self, shape: tuple[int, ...], seed: int | list[int] = 42) -> torch.Tensor:
        """Return [B, C, 1, H, W] latent noise; each item starts from its own seed, independent of batch order."""
        seeds = [seed] * shape[0] if isinstance(seed, int) else seed
        generators = [torch.Generator(self.device).manual_seed(value) for value in seeds]
        return randn_tensor(shape, generator=generators, device=self.device, dtype=self.dtype)

    def prepare_eval_inputs(
        self,
        preprocessed_data: PreprocessOutput,
        text_cfg_scale: float = 1.0,
        seed: int | list[int] = 42,
    ) -> QwenForwardOutput:
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`; source supplies size without a target.
        `text_cfg_scale` can be provided for negative_prompt encoding.
        """
        sample_mode = "argmax"

        prompt = preprocessed_data.prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, preprocessed_data.vlm_conditions)

        neg_prompt_embeds, neg_prompt_embeds_mask = None, None
        if text_cfg_scale > 1.0:
            negative_prompt = preprocessed_data.negative_prompt
            neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt(
                negative_prompt, preprocessed_data.vlm_conditions
            )

        image_shapes = []
        dit_conditions = preprocessed_data.dit_conditions
        target = preprocessed_data.target
        height, width = target.shape[-2:] if target is not None else (preprocessed_data.height, preprocessed_data.width)

        # ---------------- Encode and Pack ---------------- #
        # Encode
        noise_shape = (
            len(prompt),
            self.vae_channels,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        noise = self.sample_eval_noise(noise_shape, seed)
        conds = [self.encode_image(c, sample_mode) for c in dit_conditions.values()]
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

        return QwenForwardOutput(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_prompt_embeds_mask=neg_prompt_embeds_mask,
            noise=noise,
            conditions=cond_latents,
            height=height,
            width=width,
            image_shapes=image_shapes,
        )

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
        return self.denoise_with_transformer(
            self.transformer,
            hidden_states,
            timesteps,
            prompt_embeds,
            prompt_embeds_mask,
            img_shapes,
            img_seq_len,
            **kwargs,
        )

    def denoise_with_transformer(
        self,
        transformer: torch.nn.Module,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list[list[tuple[int]]],
        img_seq_len: int,
        **kwargs,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        predictions: torch.Tensor = transformer(
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

    def forward_step(self, batch) -> torch.Tensor:
        preprocessed_data = self.preprocess_inputs(batch)
        model_inputs = self.prepare_forward_inputs(preprocessed_data)
        hidden_states = torch.cat([model_inputs.noised_target] + [c for c in model_inputs.conditions], dim=1)
        predictions = self.denoise(
            hidden_states=hidden_states,
            timesteps=model_inputs.timesteps,
            prompt_embeds=model_inputs.prompt_embeds,
            prompt_embeds_mask=model_inputs.prompt_embeds_mask,
            img_shapes=model_inputs.image_shapes,
            img_seq_len=model_inputs.noised_target.shape[1],
        )
        loss = self.compute_loss(predictions, model_inputs.ground_truth)
        return loss

    @torch.inference_mode()
    def eval_step(
        self,
        batch: dict[str, Any],
        num_inference_steps: int = 50,
        text_cfg_scale: float = 1.0,
        seed: int | list[int] = 42,
    ) -> dict[str, torch.Tensor]:
        r"""
        Mainly used for evaluate batched data with given target images.
        """
        preprocessed_data = self.preprocess_inputs(batch)
        model_inputs = self.prepare_eval_inputs(preprocessed_data, text_cfg_scale, seed)

        xt = model_inputs.noise
        # with self.scheduler.inference_sampler(xt, num_inference_steps, xt.shape[1]) as sampler:
        with self.scheduler.inference(num_inference_steps, img_seq_len=xt.shape[1]) as inferencer:
            for t, curr_sigma, next_sigma, d_sigma_dt in tqdm(inferencer, total=num_inference_steps):
                hidden_states = torch.cat([xt] + [c for c in model_inputs.conditions], dim=1)
                timestep = t.expand(hidden_states.shape[0]).to(device=self.device, dtype=self.dtype)

                pred = self.denoise(
                    hidden_states=hidden_states,
                    timesteps=timestep,
                    prompt_embeds=model_inputs.prompt_embeds,
                    prompt_embeds_mask=model_inputs.prompt_embeds_mask,
                    img_shapes=model_inputs.image_shapes,
                    img_seq_len=xt.shape[1],
                )

                # Do CFG
                if text_cfg_scale > 1.0 and model_inputs.negative_prompt_embeds is not None:
                    neg_pred = self.denoise(
                        hidden_states=hidden_states,
                        timesteps=timestep,
                        prompt_embeds=model_inputs.negative_prompt_embeds,
                        prompt_embeds_mask=model_inputs.negative_prompt_embeds_mask,
                        img_shapes=model_inputs.image_shapes,
                        img_seq_len=xt.shape[1],
                    )
                    cfg_pred = neg_pred + text_cfg_scale * (pred - neg_pred)
                    pred_norm = torch.norm(pred, dim=-1, keepdim=True)
                    cfg_norm = torch.norm(cfg_pred, dim=-1, keepdim=True)
                    pred = (pred_norm / cfg_norm) * cfg_pred

                xt = self.scheduler.step(xt, pred, curr_sigma, next_sigma, d_sigma_dt)

        output = QwenImageEditPlusPipeline._unpack_latents(
            xt, model_inputs.height, model_inputs.width, self.vae_scale_factor
        )
        output = self.decode_image(output)
        return {"output": output}
