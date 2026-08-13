import torch
import torch.nn.functional as F
import torchvision.transforms as T

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
    CONDITION_IMAGE_SIZE,
    calculate_dimensions,
)
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Any, Literal, Optional

from schedulers import RectifiedFlowMatchingScheduler
from pipelines.base_pipeline import BasePipeline, PreprocessOutput, ForwardOutput
from data_module.utils import MAX_RESOLUTION


def resize_rgb(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize RGB tensors with antialiased Lanczos interpolation."""
    if image.shape[-2:] == (height, width):
        return image
    dtype = image.dtype
    image = T.Resize((height, width), T.InterpolationMode.BICUBIC, antialias=True)(image.float())
    return image.clamp(0, 1).to(dtype=dtype)


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
        target: torch.Tensor = batch["target"].to(self.device, dtype=self.dtype)

        negative_prompt: Optional[list[str]] = batch.get("negative_prompt", None)
        conditions: Optional[dict[str, torch.Tensor]] = batch.get("conditions", None)

        if conditions is not None:
            if isinstance(conditions, dict):
                conditions: dict[str, torch.Tensor] = {
                    k: c.to(self.device, dtype=self.dtype) for k, c in conditions.items()
                }
            else:
                conditions = [c.to(self.device, dtype=self.dtype) for k, c in conditions]

        # ---------------- Preprocess ---------------- #
        # To tensor and reshape to target areas
        h, w = target.shape[-2:]
        aspect = w / h
        w, h = calculate_dimensions(MAX_RESOLUTION, aspect)
        target = resize_rgb(target, h, w)
        target = self.image_processor.preprocess(target, h, w).unsqueeze(2)

        vlm_conditions = {}
        dit_conditions = {}
        if conditions is not None:
            for k, c in conditions.items():
                ch, cw = c.shape[-2:]
                aspect = cw / ch
                cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, aspect)
                vw, vh = calculate_dimensions(MAX_RESOLUTION, aspect)
                vlm_conditions[k] = resize_rgb(c, ch, cw)
                c = resize_rgb(c, vh, vw)
                dit_conditions[k] = self.image_processor.preprocess(c, vh, vw).unsqueeze(2)

        return PreprocessOutput(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vlm_conditions=vlm_conditions,
            dit_conditions=dit_conditions,
            target=target,
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

    def prepare_eval_inputs(
        self,
        preprocessed_data: PreprocessOutput,
        text_cfg_scale: float = 1.0,
    ) -> QwenForwardOutput:
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`, `target` must be provided.
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
        height, width = target.shape[-2:]

        # ---------------- Encode and Pack ---------------- #
        # Encode
        noise_shape = (
            target.shape[0],
            self.vae_channels,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        noise = torch.randn(noise_shape, generator=self.generator, device=self.device, dtype=self.dtype)
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
        batch,
        num_inference_steps: int = 50,
        text_cfg_scale: float = 1.0,
    ) -> list[torch.Tensor]:
        r"""
        Mainly used for evaluate batched data with given target images.
        """
        preprocessed_data = self.preprocess_inputs(batch)
        model_inputs = self.prepare_eval_inputs(preprocessed_data, text_cfg_scale)

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
