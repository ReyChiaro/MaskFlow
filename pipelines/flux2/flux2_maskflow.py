import dataclasses

import torch
import torch.nn.functional as F
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

from pipelines import maskflow_utils
from pipelines.base_pipeline import PreprocessOutput
from pipelines.flux2.flux2 import Flux2, Flux2ForwardOutput
from schedulers.flux2_flow_matching import Flux2MaskFlowScheduler


@dataclasses.dataclass
class Flux2MaskFlowPreprocessOutput(PreprocessOutput):
    raw_source: torch.Tensor | None = None
    mask: torch.Tensor | None = None


@dataclasses.dataclass
class Flux2MaskFlowForwardOutput(Flux2ForwardOutput):
    source_latents: torch.Tensor | None = None
    mask_latents: torch.Tensor | None = None
    mask_ratio: torch.Tensor | None = None


@dataclasses.dataclass
class Flux2MaskFlow(Flux2):
    scheduler: Flux2MaskFlowScheduler | None = None
    condition_keys: list[str] = dataclasses.field(default_factory=lambda: ["source", "mask"])
    mask_dilation_kernel: int = 25
    mask_blur_kernel: int = 25
    mask_blur_sigma: float = 25.0
    mask_loss_weight: float = 1.0
    enable_masked_loss: bool = True
    enable_pixel_blend: bool = True
    enable_poisson_train: bool = True
    enable_poisson_infer: bool = True
    poisson_steps: list[float] = dataclasses.field(default_factory=lambda: [0.0, 1.0])
    poisson_lambda_e: float = 1.0
    poisson_lambda_s: float = 1.0
    poisson_num_iter: int = 50
    poisson_momentum: float = 0.1

    def __post_init__(self):
        if not isinstance(self.scheduler, Flux2MaskFlowScheduler):
            raise ValueError("Flux2MaskFlow requires Flux2MaskFlowScheduler.")
        if len(self.poisson_steps) != 2 or not 0 <= self.poisson_steps[0] <= self.poisson_steps[1] <= 1:
            raise ValueError("poisson_steps must be [start, end] within [0, 1].")
        if "source" not in self.condition_keys:
            raise ValueError("MaskFlow condition_keys must include source.")
        super().__post_init__()

    def preprocess_inputs(self, batch):
        conditions = batch.get("conditions") or {}
        if "mask" not in conditions:
            return super().preprocess_inputs(batch)
        if "source" not in conditions:
            raise ValueError("MaskFlow requires a source image when a mask is supplied.")
        target = batch.get("target")
        size = self.image_size(target if target is not None else conditions["source"], self.max_area)
        source = self.resize_image(conditions["source"], size)
        mask = self.resize_image(conditions["mask"], size, is_mask=True).mean(dim=1, keepdim=True)
        if self.mask_dilation_kernel > 0:
            mask = maskflow_utils.dilate_mask(mask, self.mask_dilation_kernel)
        if self.mask_blur_kernel > 0:
            mask = maskflow_utils.blur_mask(mask, self.mask_blur_kernel, self.mask_blur_sigma)
        mask = mask.clamp(0, 1).expand_as(source)
        if target is not None:
            target = self.resize_image(target, size)
            target = self.image_processor.preprocess(target, height=size[0], width=size[1])
        # Source, mask and target must share a spatial grid for masked flow and Poisson refinement.
        spatial_conditions = dict(conditions, source=source, mask=mask)
        encoded_conditions = {}
        for key in self.condition_keys:
            if key in spatial_conditions:
                image = self.resize_image(spatial_conditions[key], size, is_mask=key == "mask")
                encoded_conditions[key] = self.image_processor.preprocess(image, height=size[0], width=size[1])
        return Flux2MaskFlowPreprocessOutput(
            prompt=batch["prompt"], negative_prompt=batch.get("negative_prompt"),
            target=target, height=size[0], width=size[1],
            dit_conditions=encoded_conditions, raw_source=source, mask=mask,
        )

    def encode_mask(self, mask):
        size = (mask.shape[-2] // self.vae_scale_factor, mask.shape[-1] // self.vae_scale_factor)
        mask = F.interpolate(mask.mean(dim=1, keepdim=True).float(), size=size, mode="nearest")
        mask = mask.repeat(1, self.vae.config.latent_channels, 1, 1).to(self.dtype)
        # Spatial weights follow patchification, but must not receive the VAE's BN normalization.
        return Flux2Pipeline._patchify_latents(mask)

    @torch.no_grad()
    def prepare_eval_inputs(self, data, text_cfg_scale=1.0):
        base = super().prepare_eval_inputs(data, text_cfg_scale)
        result = Flux2MaskFlowForwardOutput(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
        if getattr(data, "mask", None) is not None:
            result.mask_latents = Flux2Pipeline._pack_latents(self.encode_mask(data.mask))
            source_index = list(data.dit_conditions).index("source")
            result.source_latents = result.conditions[source_index]
            result.mask_ratio = data.mask.flatten(1).mean(dim=1).view(-1, 1, 1)
            result.noised_target = self.scheduler.add_noise_by_sigmas(
                result.noise, result.source_latents, torch.ones(1, device=self.device),
                result.source_latents, result.mask_latents,
            )
        return result

    def unpack_spatial(self, tokens, ids):
        return Flux2Pipeline._unpatchify_latents(Flux2Pipeline._unpack_latents_with_ids(tokens, ids))

    def refine_target(self, target, inputs):
        source = self.unpack_spatial(inputs.source_latents, inputs.latent_ids)
        mask = self.unpack_spatial(inputs.mask_latents, inputs.latent_ids)
        image = Flux2Pipeline._unpatchify_latents(target)
        image = maskflow_utils.poisson_refine(
            image, source, mask >= 0.5, mask, self.poisson_lambda_e, self.poisson_lambda_s,
            self.poisson_num_iter, self.poisson_momentum, disable_progress_bar=True,
        )
        return Flux2Pipeline._patchify_latents(image).to(target)

    @torch.no_grad()
    def prepare_forward_inputs(self, data):
        if data.target is None:
            raise ValueError("Training requires target images.")
        if getattr(data, "mask", None) is None:
            return super().prepare_forward_inputs(data)
        result = self.prepare_eval_inputs(data)
        target = self.encode_image(data.target, self.train_sample_mode)
        if self.enable_poisson_train:
            target = self.refine_target(target, result)
        target = Flux2Pipeline._pack_latents(target)
        ts = self.scheduler.sample_timesteps(target.shape[0], self.generator, self.device)
        result.sigmas = self.scheduler.get_sigmas(ts, target.shape[1])
        result.timesteps = result.sigmas
        result.noised_target = self.scheduler.add_noise_by_sigmas(
            result.noise, target, result.sigmas, result.source_latents, result.mask_latents,
        )
        result.ground_truth = self.scheduler.get_velocity(result.noise, target, result.source_latents, result.mask_latents)
        return result

    def compute_loss(self, prediction, inputs):
        if not self.enable_masked_loss or inputs.mask_latents is None:
            return super().compute_loss(prediction, inputs)
        loss = F.mse_loss(prediction.float(), inputs.ground_truth.float(), reduction="none")
        loss = self.mask_loss_weight * inputs.mask_latents * loss / (inputs.mask_ratio + 1e-6)
        return {"loss": loss.flatten(1).mean(dim=1).mean(), "mask_ratio": inputs.mask_ratio.mean()}

    def inference_step(self, xt, prediction, sigma, next_sigma, derivative, inputs):
        if inputs.mask_latents is None:
            return super().inference_step(xt, prediction, sigma, next_sigma, derivative, inputs)
        if self.enable_poisson_infer and self.poisson_steps[0] <= sigma.item() <= self.poisson_steps[1]:
            x0 = self.scheduler.predict_x0(xt, sigma, derivative, prediction)
            x0 = Flux2Pipeline._unpack_latents_with_ids(x0, inputs.latent_ids)
            refined = Flux2Pipeline._pack_latents(self.refine_target(x0, inputs))
            prediction = (xt - refined) / sigma.clamp_min(1e-4)
        return self.scheduler.step(
            xt, prediction, sigma, next_sigma, derivative, inputs.source_latents, inputs.mask_latents, inputs.noise,
        )

    def postprocess_output(self, output, data):
        if getattr(data, "mask", None) is None:
            return super().postprocess_output(output, data)
        if self.enable_pixel_blend:
            output = data.mask * output + (1 - data.mask) * data.raw_source
        return {"output": output, "mask": data.mask}
