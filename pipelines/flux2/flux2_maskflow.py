import dataclasses

import torch
import torch.nn.functional as F
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

from pipelines import maskflow_utils
from pipelines.base_pipeline import PreprocessOutput
from pipelines.cfg import branch_conditions, combine_predictions, required_branches
from pipelines.flux2.flux2 import Flux2, Flux2ForwardOutput
from schedulers.flux2_flow_matching import Flux2MaskFlowScheduler


@dataclasses.dataclass
class Flux2MaskFlowPreprocessOutput(PreprocessOutput):
    raw_source: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    cfg_branch: str = "pm"


@dataclasses.dataclass
class Flux2MaskFlowForwardOutput(Flux2ForwardOutput):
    source_latents: torch.Tensor | None = None
    mask_latents: torch.Tensor | None = None
    mask_ratio: torch.Tensor | None = None
    cfg_branch: str = "pm"
    cfg_branches: dict[str, Flux2ForwardOutput] = dataclasses.field(default_factory=dict)


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
        if not {"source", "mask"}.issubset(self.condition_keys):
            raise ValueError("MaskFlow condition_keys must include source and mask.")
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
            prompt=batch["prompt"],
            negative_prompt=batch.get("negative_prompt"),
            target=target,
            height=size[0],
            width=size[1],
            dit_conditions=encoded_conditions,
            raw_source=source,
            mask=mask,
            cfg_branch=batch.get("cfg_branch", "pm"),
        )

    def encode_mask(self, mask):
        size = (mask.shape[-2] // self.vae_scale_factor, mask.shape[-1] // self.vae_scale_factor)
        mask = F.interpolate(mask.mean(dim=1, keepdim=True).float(), size=size, mode="nearest")
        mask = mask.repeat(1, self.vae.config.latent_channels, 1, 1).to(self.dtype)
        # Spatial weights follow patchification, but must not receive the VAE's BN normalization.
        return Flux2Pipeline._patchify_latents(mask)

    @torch.no_grad()
    def _prepare_mask_inputs(
        self,
        data: Flux2MaskFlowPreprocessOutput,
        training: bool = False,
        seed: int | list[int] | None = None,
    ) -> Flux2MaskFlowForwardOutput:
        # Encode all physical conditions once, independently of condition dropout.
        # All ranks must make exactly one text-encoder call during training,
        # including when that encoder is sharded with FSDP.
        encode_data = data
        if training and not branch_conditions(data.cfg_branch)[0]:
            encode_data = dataclasses.replace(data, prompt=[""] * len(data.prompt))
        base = super().prepare_eval_inputs(encode_data, seed=seed)
        result = Flux2MaskFlowForwardOutput(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
        if getattr(data, "mask", None) is not None:
            result.mask_latents = Flux2Pipeline._pack_latents(self.encode_mask(data.mask))
            source_index = list(data.dit_conditions).index("source")
            result.source_latents = result.conditions[source_index]
            result.mask_ratio = data.mask.flatten(1).mean(dim=1).view(-1, 1, 1)
            result.noised_target = self.scheduler.add_noise_by_sigmas(
                result.noise,
                result.source_latents,
                torch.ones(1, device=self.device),
                result.source_latents,
                result.mask_latents,
            )
        return result

    def build_cfg_branch(self, data, name, inputs, training=False, prompt_cache=None):
        keep_text, keep_mask = branch_conditions(name)
        if prompt_cache is None:
            prompt_cache = {True: (inputs.prompt_embeds, inputs.text_ids)}
        if keep_text not in prompt_cache:
            prompt = [""] * len(data.prompt) if training else data.negative_prompt or [""] * len(data.prompt)
            prompt_cache[keep_text] = self.encode_prompt(prompt)
        prompt_embeds, text_ids = prompt_cache[keep_text]
        conditions = []
        image_ids = [inputs.latent_ids]
        offset = inputs.noise.shape[1]
        for key, condition in zip(data.dit_conditions, inputs.conditions):
            end = offset + condition.shape[1]
            if keep_mask or key != "mask":
                conditions.append(condition)
                # Preserve reference coordinates while removing exactly the mask tokens.
                image_ids.append(inputs.image_ids[:, offset:end])
            offset = end
        return Flux2ForwardOutput(
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            conditions=conditions,
            image_ids=torch.cat(image_ids, dim=1),
            latent_ids=inputs.latent_ids,
        )

    @torch.no_grad()
    def prepare_eval_inputs(
        self,
        data: Flux2MaskFlowPreprocessOutput,
        text_cfg_scale: float = 1.0,
        mask_cfg_scale: float = 1.0,
        interaction_cfg_scale: float | None = None,
        seed: int | list[int] | None = None,
    ) -> Flux2ForwardOutput:
        if getattr(data, "mask", None) is None:
            if mask_cfg_scale != 1.0 or interaction_cfg_scale is not None:
                raise ValueError("Mask CFG requires source and mask inputs.")
            return super().prepare_eval_inputs(data, text_cfg_scale, seed=seed)
        inputs = self._prepare_mask_inputs(data, seed=seed)
        prompt_cache = {True: (inputs.prompt_embeds, inputs.text_ids)}
        inputs.cfg_branches = {
            name: self.build_cfg_branch(data, name, inputs, prompt_cache=prompt_cache)
            for name in required_branches(text_cfg_scale, mask_cfg_scale, interaction_cfg_scale, self.rescale_cfg)
        }
        return inputs

    def combine_cfg_predictions(self, predictions, text_cfg_scale=1.0, mask_cfg_scale=1.0, interaction_cfg_scale=None):
        return combine_predictions(predictions, text_cfg_scale, mask_cfg_scale, interaction_cfg_scale, self.rescale_cfg)

    def predict_velocity(self, xt, timestep, inputs, text_cfg_scale, mask_cfg_scale=1.0, interaction_cfg_scale=None):
        if not getattr(inputs, "cfg_branches", None):
            return super().predict_velocity(xt, timestep, inputs, text_cfg_scale)
        predictions = {name: self.denoise(xt, timestep, branch) for name, branch in inputs.cfg_branches.items()}
        return self.combine_cfg_predictions(predictions, text_cfg_scale, mask_cfg_scale, interaction_cfg_scale)

    def unpack_spatial(self, tokens, ids):
        return Flux2Pipeline._unpatchify_latents(Flux2Pipeline._unpack_latents_with_ids(tokens, ids))

    def refine_target(self, target, inputs):
        source = self.unpack_spatial(inputs.source_latents, inputs.latent_ids)
        mask = self.unpack_spatial(inputs.mask_latents, inputs.latent_ids)
        image = Flux2Pipeline._unpatchify_latents(target)
        image = maskflow_utils.poisson_refine(
            image,
            source,
            mask > 0,
            mask,
            self.poisson_lambda_e,
            self.poisson_lambda_s,
            self.poisson_num_iter,
            self.poisson_momentum,
            disable_progress_bar=True,
        )
        return Flux2Pipeline._patchify_latents(image).to(target)

    @torch.no_grad()
    def prepare_forward_inputs(self, data):
        if data.target is None:
            raise ValueError("Training requires target images.")
        if getattr(data, "mask", None) is None:
            return super().prepare_forward_inputs(data)
        result = self._prepare_mask_inputs(data, training=True)
        target = self.encode_image(data.target, self.train_sample_mode)
        if self.enable_poisson_train:
            target = self.refine_target(target, result)
        target = Flux2Pipeline._pack_latents(target)
        ts = self.scheduler.sample_timesteps(target.shape[0], self.generator, self.device)
        result.sigmas = self.scheduler.get_sigmas(ts, target.shape[1])
        result.timesteps = result.sigmas
        result.noised_target = self.scheduler.add_noise_by_sigmas(
            result.noise,
            target,
            result.sigmas,
            result.source_latents,
            result.mask_latents,
        )
        result.ground_truth = self.scheduler.get_velocity(
            result.noise, target, result.source_latents, result.mask_latents, sigmas=result.sigmas
        )
        keep_text, _ = branch_conditions(data.cfg_branch)
        branch = self.build_cfg_branch(
            data,
            data.cfg_branch,
            result,
            training=True,
            prompt_cache={keep_text: (result.prompt_embeds, result.text_ids)},
        )
        result.prompt_embeds = branch.prompt_embeds
        result.text_ids = branch.text_ids
        result.conditions = branch.conditions
        result.image_ids = branch.image_ids
        result.cfg_branch = data.cfg_branch
        return result

    def compute_loss(self, prediction, inputs):
        if (
            not self.enable_masked_loss
            or getattr(inputs, "mask_latents", None) is None
            or not branch_conditions(inputs.cfg_branch)[1]
        ):
            return super().compute_loss(prediction, inputs)
        loss = F.mse_loss(prediction.float(), inputs.ground_truth.float(), reduction="none")
        loss = self.mask_loss_weight * inputs.mask_latents * loss / (inputs.mask_ratio + 1e-6)
        return {"loss": loss.flatten(1).mean(dim=1).mean(), "mask_ratio": inputs.mask_ratio.mean()}

    def inference_step(self, xt, prediction, sigma, next_sigma, derivative, inputs):
        if inputs.mask_latents is None:
            return super().inference_step(xt, prediction, sigma, next_sigma, derivative, inputs)
        if self.enable_poisson_infer and self.poisson_steps[0] <= sigma.item() <= self.poisson_steps[1]:
            x0_pred = self.scheduler.predict_x0(
                xt.float(), sigma, derivative, prediction,
                inputs.source_latents, inputs.mask_latents, inputs.noise,
            )
            x0 = Flux2Pipeline._unpack_latents_with_ids(x0_pred.to(xt.dtype), inputs.latent_ids)
            refined = Flux2Pipeline._pack_latents(self.refine_target(x0, inputs))
            if self.scheduler.unmask_with != "noisy_source" or self.scheduler.background_noise_power == 1.0:
                prediction = (xt - refined) / sigma.clamp_min(1e-4)
            else:
                # The nonlinear path correction cancels between the two clean estimates.
                prediction = prediction.float() + (x0_pred - refined.float()) / sigma.clamp_min(1e-4)
        return self.scheduler.step(
            xt,
            prediction,
            sigma,
            next_sigma,
            derivative,
            inputs.source_latents,
            inputs.mask_latents,
            inputs.noise,
        )

    def postprocess_output(self, output, data):
        if getattr(data, "mask", None) is None:
            return super().postprocess_output(output, data)
        if self.enable_pixel_blend:
            output = data.mask * output + (1 - data.mask) * data.raw_source
        return {"output": output, "mask": data.mask}
