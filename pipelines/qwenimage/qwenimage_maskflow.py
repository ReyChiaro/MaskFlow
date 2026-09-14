import copy
import dataclasses
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
)
from loguru import logger
from PIL import Image
from tqdm import tqdm

from data_module.utils import image_dimensions, preprocess_images
from pipelines import maskflow_utils
from pipelines.base_pipeline import PreprocessOutput
from pipelines.cfg import (
    branch_conditions,
    combine_predictions,
    required_branches,
)
from pipelines.qwenimage.qwenimage_edit_plus import (
    QwenForwardOutput,
    QwenImageEditPlus,
    resize_rgb,
)
from schedulers import MaskFlowScheduler


@dataclasses.dataclass
class QwenMaskFlowPreprocessOutput(PreprocessOutput):
    raw_source: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    cfg_branch: str = "pm"


@dataclasses.dataclass
class QwenMaskFlowCFGBranch:
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    conditions: list[torch.Tensor]
    image_shapes: list


@dataclasses.dataclass
class QwenMaskFlowForwardOutput(QwenForwardOutput):
    r"""
    A wrapper for training and inference.
    Note that we do not use `cfg_branches` at train stage and specify `prompt_embeds` etc. clearly,
    while at inference stage, we recommand use `cfg_branches` to define different `prompt_embeds` for
    different denoise branches.
    """

    loss_weights: torch.Tensor | None = None
    source_latents: torch.Tensor | None = None
    cfg_branch: str = "pm"

    # Mask and edge latents that are ``encoded'' by interpolation rather than VAE.
    mask_latents: torch.Tensor | None = None
    mask_ratio: torch.Tensor | None = None

    # We recommand use `cfg_branches` to specify different `prompt_embeds` and `prompt_embeds_mask`
    # while ignore the original attributes.
    cfg_branches: dict[str, QwenMaskFlowCFGBranch] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class QwenImageMaskFlow(QwenImageEditPlus):
    r"""
    Arg
        mask_loss_weight   (float): Weight for mask area-adaptive loss
        edge_loss_weight   (float): Weight for edge loss

        enable_masked_loss (bool): Whether to use Full vector fields prediction or Masked/Edged vector fields
        mask_denoise_steps (tuple[int|float]|int|float): Predict masked vector fields between steps [start, end]
            including both start point and end point. If the start is not given, use 0 by default.
        mask_denoise_train (bool): Predict masked vector fields when training.
        mask_denoise_infer (bool): Predict masked vector fields when inferring.

        enable_pixel_blend (bool): Enable the final pixel space blend with source image after VAE decoding.
    """

    scheduler: MaskFlowScheduler | None = None

    mask_dilation_kernel: int = 25
    mask_blur_kernel: int = 25
    mask_blur_sigma: float = 25.0

    mask_loss_weight: float = 1.0
    enable_masked_loss: bool = True
    rescale_cfg: bool = True
    enable_pixel_blend: bool = True

    enable_poisson_train: bool = True
    enable_poisson_infer: bool = True
    poisson_steps: list[float] = dataclasses.field(default_factory=list)
    poisson_lambda_e: float = 1.0
    poisson_lambda_s: float = 1.0
    poisson_num_iter: int = 50
    poisson_momentum: float = 0.1

    def __post_init__(self):
        super().__post_init__()

    # ---------------- Mask Encoding ---------------- #
    def encode_mask(self, mask: torch.Tensor):
        r"""
        Mimic VAE for mask, conduct 8x downsample on mask and pack to 3D tensor.

        Args:
            mask (Tensor): [B, C, H, W], without the frame dimension.
        """
        h, w = mask.shape[-2:]
        mask = F.interpolate(mask, [h // self.vae_scale_factor, w // self.vae_scale_factor], mode="nearest")
        mask = mask.mean(dim=1, keepdim=True).repeat(1, self.vae_channels, 1, 1).unsqueeze(2)
        return mask.clamp(0, 1)

    # ---------------- Pipeline Operations ---------------- #
    def preprocess_inputs(self, batch: dict[str, Any]) -> QwenMaskFlowPreprocessOutput:
        r"""
        A batched data is supposed to have keys `prompt`, `target`.
        """
        prompt: list[str] = batch["prompt"]
        target = batch.get("target")
        if target is not None:
            target = target.to(self.device, dtype=self.dtype)

        negative_prompt: Optional[list[str]] = batch.get("negative_prompt", None)
        conditions: Optional[dict[str, torch.Tensor]] = batch.get("conditions", None)

        if conditions is None or "mask" not in conditions or "source" not in conditions:
            logger.warning(
                "QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().preprocess_inputs(batch)

        # Spatial preprocessing has already produced aligned CPU float32 [B, C, H, W] tensors in [0, 1].
        source = conditions["source"]
        mask = conditions["mask"].to(self.device, dtype=self.dtype)
        h, w = source.shape[-2:]
        ch, cw = image_dimensions(w / h, self.max_condition_resolution, self.divisible_by)
        vlm_source = resize_rgb(source, ch, cw).to(self.device, dtype=self.dtype)
        source = source.to(self.device, dtype=self.dtype)

        # Morphology is applied once at the shared model resolution.
        if self.mask_dilation_kernel > 0:
            mask = maskflow_utils.dilate_mask(mask, self.mask_dilation_kernel)
        if self.mask_blur_kernel > 0:
            mask = maskflow_utils.blur_mask(mask, self.mask_blur_kernel, self.mask_blur_sigma)
        if target is not None:
            target = self.image_processor.preprocess(target, h, w).unsqueeze(2)
        vlm_conditions = {
            "source": vlm_source,
            "mask": F.interpolate(mask.float(), size=(ch, cw), mode="nearest").to(dtype=mask.dtype),
        }
        dit_conditions = {
            "source": self.image_processor.preprocess(source, h, w).unsqueeze(2),
            "mask": self.image_processor.preprocess(mask, h, w).unsqueeze(2),
        }
        raw_source = source

        return QwenMaskFlowPreprocessOutput(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vlm_conditions=vlm_conditions,
            dit_conditions=dit_conditions,
            target=target,
            height=h,
            width=w,
            raw_source=raw_source,
            mask=mask,
            cfg_branch=batch.get("cfg_branch", "pm"),
        )

    def build_cfg_branch(
        self,
        data: QwenMaskFlowPreprocessOutput,
        name: str,
        conditions: list[torch.Tensor],
        image_shapes: list[tuple[int]],
        training: bool = False,
    ):
        keep_text, keep_mask = branch_conditions(name)
        prompt = (
            data.prompt
            if keep_text
            else ([""] * len(data.prompt) if training else data.negative_prompt or [""] * len(data.prompt))
        )
        vlm_conditions = {key: value for key, value in data.vlm_conditions.items() if keep_mask or key != "mask"}
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, vlm_conditions)
        # The image shape list contains the generated image, source, then mask.
        return QwenMaskFlowCFGBranch(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            conditions=conditions if keep_mask else conditions[:1],
            image_shapes=image_shapes if keep_mask else [shapes[:2] for shapes in image_shapes],
        )

    def prepare_forward_inputs(self, preprocessed_data: QwenMaskFlowPreprocessOutput) -> QwenMaskFlowForwardOutput:
        r"""
        Prepare training forward inputs.
        The sample mode for VAE is fixed to `sample`, `target` must be provided.
        """
        if preprocessed_data.target is None:
            raise ValueError("Training requires target images.")
        sample_mode = "sample"

        if getattr(preprocessed_data, "mask", None) is None:
            logger.warning(
                "QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().prepare_forward_inputs(preprocessed_data)

        dit_conditions = preprocessed_data.dit_conditions

        image_shapes = []
        target = preprocessed_data.target
        mask = preprocessed_data.mask
        height, width = target.shape[-2:]

        # Mask ratio mast be calculated before the masks being encoded for pixel-aligned
        mask_ratio = mask.flatten(1).sum(dim=-1, keepdim=True) / math.prod(mask.shape[1:])
        mask_ratio = mask_ratio.view(mask.shape[0], 1, 1)

        # ---------------- Encode Images and Pack ---------------- #
        tgt = self.encode_image(target, sample_mode)  # Encode target
        conds = [
            self.encode_image(dit_conditions["source"], sample_mode),
            self.encode_image(dit_conditions["mask"], sample_mode),
        ]  # Encode conditions including the masks with VAE

        # This reshaped mask latents will be used for regional controlling
        mask_latents = self.encode_mask(mask)

        # Conducting Poisson refinement on the target:
        # paste the backgrounds from source images to the targets
        if self.enable_poisson_train:
            tgt_dtype = tgt.dtype
            tgt = maskflow_utils.poisson_refine(
                g=tgt.squeeze(2),
                x_S=conds[0].squeeze(2),
                M=mask_latents.squeeze(2) > 0,  # 1 for masked regions to be edited
                soft_M=mask_latents.squeeze(2),
                poisson_lambda_e=self.poisson_lambda_e,
                poisson_lambda_s=self.poisson_lambda_s,
                poisson_num_iter=self.poisson_num_iter,
                poisson_momentum=self.poisson_momentum,
                disable_progress_bar=True,
            )
            tgt = tgt.unsqueeze(2).to(dtype=tgt_dtype)

        image_shapes.append((1, tgt.shape[-2] // self.pacth_size, tgt.shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * tgt.shape[0]

        # Pack to 3D
        tgt: torch.Tensor = QwenImageEditPlusPipeline._pack_latents(
            tgt, tgt.shape[0], tgt.shape[1], tgt.shape[-2], tgt.shape[-1]
        )
        cond_latents: list[torch.Tensor] = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]

        # Pack mask to 3D to satisfy the shapes of target and conditions.
        mask_latents: torch.Tensor = QwenImageEditPlusPipeline._pack_latents(
            mask_latents, mask_latents.shape[0], mask_latents.shape[1], mask_latents.shape[-2], mask_latents.shape[-1]
        )

        # --------------- Sample and Add Noise -------------- #
        source = cond_latents[0]
        noise = torch.randn(tgt.shape, generator=self.generator, device=tgt.device, dtype=tgt.dtype)
        ts = self.scheduler.sample_timesteps(tgt.shape[0], self.generator, self.device)
        sigmas = self.scheduler.get_sigmas(ts, img_seq_len=tgt.shape[1])

        xt = self.scheduler.add_noise_by_sigmas(noise, tgt, sigmas, source, mask_latents)
        gt = self.scheduler.get_velocity(noise, tgt, source, mask_latents)

        branch = self.build_cfg_branch(
            preprocessed_data, preprocessed_data.cfg_branch, cond_latents, image_shapes, training=True
        )
        return QwenMaskFlowForwardOutput(
            prompt_embeds=branch.prompt_embeds,
            prompt_embeds_mask=branch.prompt_embeds_mask,
            height=height,
            width=width,
            noise=noise,
            noised_target=xt,
            timesteps=sigmas,  # Qwen use sigma-based denoise
            sigmas=sigmas,
            ground_truth=gt,
            conditions=branch.conditions,
            image_shapes=branch.image_shapes,
            source_latents=source,
            cfg_branch=preprocessed_data.cfg_branch,
            mask_latents=mask_latents,
            mask_ratio=mask_ratio,
        )

    def prepare_eval_inputs(
        self,
        preprocessed_data: QwenMaskFlowPreprocessOutput,
        text_cfg_scale: float = 1.0,
        mask_cfg_scale: float = 1.0,
        interaction_cfg_scale: float | None = None,
        seed: int | list[int] = 42,
    ) -> QwenMaskFlowForwardOutput:
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`; source supplies size without a target.
        """
        sample_mode = "argmax"

        if getattr(preprocessed_data, "mask", None) is None:
            logger.warning(
                "QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().prepare_eval_inputs(preprocessed_data, text_cfg_scale, seed)

        prompt = preprocessed_data.prompt

        image_shapes = []
        dit_conditions = preprocessed_data.dit_conditions
        target = preprocessed_data.target
        mask = preprocessed_data.mask
        height, width = target.shape[-2:] if target is not None else (preprocessed_data.height, preprocessed_data.width)

        # ---------------- Encode and Pack ---------------- #
        noise_shape = (
            len(prompt),
            self.vae_channels,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        noise = self.sample_eval_noise(noise_shape, seed)
        conds = [
            self.encode_image(dit_conditions["source"], sample_mode),
            self.encode_image(dit_conditions["mask"], sample_mode),
        ]
        mask_latents = self.encode_mask(mask)

        image_shapes.append((1, noise_shape[-2] // self.pacth_size, noise_shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * noise.shape[0]
        # Pack to 3D
        noise = QwenImageEditPlusPipeline._pack_latents(
            noise, noise_shape[0], noise_shape[1], noise_shape[-2], noise_shape[-1]
        )
        cond_latents = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]
        mask_latents = QwenImageEditPlusPipeline._pack_latents(
            mask_latents, mask_latents.shape[0], mask_latents.shape[1], mask_latents.shape[-2], mask_latents.shape[-1]
        )

        cfg_branches = {
            name: self.build_cfg_branch(preprocessed_data, name, cond_latents, image_shapes)
            for name in required_branches(text_cfg_scale, mask_cfg_scale, interaction_cfg_scale, self.rescale_cfg)
        }
        return QwenMaskFlowForwardOutput(
            height=height,
            width=width,
            noise=noise,
            conditions=cond_latents,
            source_latents=cond_latents[0],
            mask_latents=mask_latents,
            cfg_branches=cfg_branches,
        )

    @torch.inference_mode()
    def denoise_cfg_branch(
        self,
        branch: QwenMaskFlowCFGBranch,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        A CFG branch wrapper for denoise with `inference_mode` decorator.
        """
        hidden_states = torch.cat([xt] + branch.conditions, dim=1)
        return self.denoise(
            hidden_states=hidden_states,
            timesteps=timestep,
            prompt_embeds=branch.prompt_embeds,
            prompt_embeds_mask=branch.prompt_embeds_mask,
            img_shapes=branch.image_shapes,
            img_seq_len=xt.shape[1],
        )

    def compute_loss(
        self,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        mask_latents: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Compute loss with masks and edges.

        Args
            predictions   (Tensor): The model predicted vector fields.
            ground_truths (Tensor): The ground truth vector fields.
            mask_ratio    (Tensor): Masked area / image area.
                Used to zoom the masked loss weights adaptively during training.
            mask_latents  (Tensor): The reshaped/VAE encoded softened binary masks.

        Return
            dict[str, Tensor]: The dict of different types losses.
        """
        loss_dict = {}
        loss_field = F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
        loss = None

        if self.enable_masked_loss and mask_latents is not None:
            mask_field = mask_latents * loss_field

            if mask_ratio is not None:
                mask_field = (1.0 / (mask_ratio + 1e-6)) * mask_field
                loss_dict["mask_ratio"] = mask_ratio.mean()

            loss_field = loss_field + self.mask_loss_weight * mask_field

        loss = (loss_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
        loss_dict["loss"] = loss
        return loss_dict


    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        # `preprocess_inputs` includes extracting prompts and images from batched samples
        # and resize or conduct other augmentation on them.
        preprocessed_data = self.preprocess_inputs(batch)

        if getattr(preprocessed_data, "mask", None) is None:
            return super().forward_step(batch)

        # `prepare_forward_inputs` encodes all prompts and images into latents,
        # `MaskFlow` will additionally conduct Poisson replace on the target images.
        model_inputs = self.prepare_forward_inputs(preprocessed_data)

        # One-time step. Note that Qwen use sigma-based denoise.
        hidden_states = torch.cat([model_inputs.noised_target] + [c for c in model_inputs.conditions], dim=1)
        predictions = self.denoise(
            hidden_states=hidden_states,
            timesteps=model_inputs.timesteps,
            prompt_embeds=model_inputs.prompt_embeds,
            prompt_embeds_mask=model_inputs.prompt_embeds_mask,
            img_shapes=model_inputs.image_shapes,
            img_seq_len=model_inputs.noised_target.shape[1],
        )
        loss = self.compute_loss(
            predictions=predictions,
            ground_truths=model_inputs.ground_truth,
            mask_ratio=model_inputs.mask_ratio,
            mask_latents=(model_inputs.mask_latents if branch_conditions(model_inputs.cfg_branch)[1] else None),
        )
        return loss

    def apply_poisson_to_prediction(
        self,
        xt: torch.Tensor,
        pred: torch.Tensor,
        curr_sigma: torch.Tensor,
        d_sigma_dt: torch.Tensor,
        source: torch.Tensor,
        mask_latents: torch.Tensor,
        noise: torch.Tensor,
        height: int,
        width: int,
        disable_progress_bar: bool = False,
    ) -> torch.Tensor:
        x_S: torch.Tensor = QwenImageEditPlusPipeline._unpack_latents(
            source, height, width, self.vae_scale_factor
        ).squeeze(2)
        soft_M: torch.Tensor = QwenImageEditPlusPipeline._unpack_latents(
            mask_latents, height, width, self.vae_scale_factor
        ).squeeze(2)

        x0_pred = self.scheduler.predict_x0(xt, curr_sigma, d_sigma_dt, pred, source, mask_latents, noise)
        x0_dtype = x0_pred.dtype
        x0_pred = QwenImageEditPlusPipeline._unpack_latents(x0_pred, height, width, self.vae_scale_factor).squeeze(2)
        x0_refined = maskflow_utils.poisson_refine(
            g=x0_pred,
            x_S=x_S,
            M=soft_M > 0,
            soft_M=soft_M,
            poisson_lambda_e=self.poisson_lambda_e,
            poisson_lambda_s=self.poisson_lambda_s,
            poisson_num_iter=self.poisson_num_iter,
            poisson_momentum=self.poisson_momentum,
            disable_progress_bar=disable_progress_bar,
        )
        x0_refined = QwenImageEditPlusPipeline._pack_latents(
            x0_refined,
            x0_refined.shape[0],
            x0_refined.shape[1],
            x0_refined.shape[-2],
            x0_refined.shape[-1],
        ).to(dtype=x0_dtype)
        while curr_sigma.ndim < xt.ndim:
            curr_sigma = curr_sigma.unsqueeze(-1)
        return (xt - x0_refined) / curr_sigma.clamp_min(1e-4)

    @torch.inference_mode()
    def eval_step(
        self,
        batch: dict[str, Any],
        num_inference_steps: int = 50,
        text_cfg_scale: float = 1.0,
        mask_cfg_scale: float = 1.0,
        interaction_cfg_scale: float | None = None,
        seed: int | list[int] = 42,
    ) -> dict[str, torch.Tensor]:
        r"""
        Evaluation step is used for batched sample evaluation, especially on benchmarks or testsets.
        """
        # As the batch share the same structure with training batch, we use the same extraction method as train.
        preprocessed_data = self.preprocess_inputs(batch)
        if getattr(preprocessed_data, "mask", None) is None:
            if mask_cfg_scale != 1.0 or interaction_cfg_scale is not None:
                raise ValueError("Mask CFG requires source and mask inputs.")
            return super().eval_step(batch, num_inference_steps, text_cfg_scale, seed)
        model_inputs = self.prepare_eval_inputs(
            preprocessed_data, text_cfg_scale, mask_cfg_scale, interaction_cfg_scale, seed
        )
        xt = model_inputs.noise
        mask = preprocessed_data.mask
        raw_source = preprocessed_data.raw_source
        source = model_inputs.source_latents
        mask_latents = model_inputs.mask_latents
        noise = copy.deepcopy(model_inputs.noise)

        with self.scheduler.inference(num_inference_steps, img_seq_len=xt.shape[1]) as inferencer:
            for step_index, (t, curr_sigma, next_sigma, d_sigma_dt) in enumerate(
                tqdm(inferencer, total=num_inference_steps), start=1
            ):
                curr_sigma = curr_sigma.to(xt.device)
                next_sigma = next_sigma.to(xt.device)
                d_sigma_dt = d_sigma_dt.to(xt.device)

                timestep = t.expand(xt.shape[0]).to(device=self.device, dtype=self.dtype)
                cfg_branches = model_inputs.cfg_branches

                branch_preds = {}
                for k, branch in cfg_branches.items():
                    branch_preds[k] = self.denoise_cfg_branch(branch, xt, timestep)
                pred = combine_predictions(
                    branch_preds, text_cfg_scale, mask_cfg_scale, interaction_cfg_scale, self.rescale_cfg
                )

                # Optionally handle Poisson refinement
                if self.enable_poisson_infer and self.poisson_steps[0] <= t.item() <= self.poisson_steps[1]:
                    pred = self.apply_poisson_to_prediction(
                        xt=xt,
                        pred=pred,
                        curr_sigma=curr_sigma,
                        d_sigma_dt=d_sigma_dt,
                        source=source,
                        mask_latents=mask_latents,
                        noise=noise,
                        height=model_inputs.height,
                        width=model_inputs.width,
                        disable_progress_bar=True,
                    )

                # runtime_mask = self.inference_mask(timestep, mask_latents)
                xt = self.scheduler.step(xt, pred, curr_sigma, next_sigma, d_sigma_dt, source, mask_latents, noise)

        output = QwenImageEditPlusPipeline._unpack_latents(
            xt, model_inputs.height, model_inputs.width, self.vae_scale_factor
        )
        output = self.decode_image(output)

        if self.enable_pixel_blend:
            output = mask * output + (1.0 - mask) * raw_source

        return {"mask": mask, "output": output}

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        source_image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
        max_resolution: int = 1024 * 1024,
        divisible_by: int = 32,
        aspect_ratios: tuple[str, ...] = (
            "1:1",
            "1:4",
            "1:8",
            "2:3",
            "3:2",
            "3:4",
            "4:1",
            "4:3",
            "4:5",
            "5:4",
            "8:1",
            "9:16",
            "16:9",
            "21:9",
        ),
        num_inference_steps: int = 50,
        text_cfg_scale: float = 4.0,
        mask_dilation_kernel: int = 25,
        mask_blur_kernel: int = 25,
        mask_blur_sigma: float = 25.0,
        enable_pixel_blend: bool = True,
        enable_poisson_refine: bool = True,
        poisson_lambda_e: float = 1.0,
        poisson_lambda_s: float = 1.0,
        poisson_momentum: float = 0.1,
        mask_cfg_scale: float = 1.0,
        interaction_cfg_scale: float | None = None,
        seed: int | list[int] = 42,
    ) -> dict[str, torch.Tensor]:
        if prompt is None or source_image is None or mask_image is None:
            raise ValueError("MaskFlow generation requires prompt, source_image and mask_image.")
        source_image = source_image.convert("RGB")
        mask_image = mask_image.convert("RGB")
        conditions, _ = preprocess_images(
            {"source": T.to_tensor(source_image), "mask": T.to_tensor(mask_image)},
            max_resolution=max_resolution,
            divisible_by=divisible_by,
            aspect_ratios=aspect_ratios,
        )
        batch = {
            "prompt": [prompt],
            "negative_prompt": [negative_prompt or ""],
            "conditions": {key: image.unsqueeze(0) for key, image in conditions.items()},
        }
        options = dict(
            mask_dilation_kernel=mask_dilation_kernel,
            mask_blur_kernel=mask_blur_kernel,
            mask_blur_sigma=mask_blur_sigma,
            enable_pixel_blend=enable_pixel_blend,
            enable_poisson_infer=enable_poisson_refine,
            poisson_lambda_e=poisson_lambda_e,
            poisson_lambda_s=poisson_lambda_s,
            poisson_momentum=poisson_momentum,
        )
        previous = {key: getattr(self, key) for key in options}
        try:
            for key, value in options.items():
                setattr(self, key, value)
            return self.eval_step(
                batch, num_inference_steps, text_cfg_scale, mask_cfg_scale, interaction_cfg_scale, seed
            )
        finally:
            for key, value in previous.items():
                setattr(self, key, value)
