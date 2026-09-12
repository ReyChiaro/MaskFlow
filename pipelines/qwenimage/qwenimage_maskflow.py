import copy
import math
import torch
import torch.nn.functional as F
import dataclasses
import torchvision.transforms.functional as T

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
    CONDITION_IMAGE_SIZE,
    calculate_dimensions,
)
from PIL import Image
from tqdm import tqdm
from typing import Any, Optional
from loguru import logger

from schedulers import MaskFlowScheduler
from pipelines import maskflow_utils
from pipelines.base_pipeline import PreprocessOutput
from pipelines.qwenimage.qwenimage_edit_plus import QwenImageEditPlus, QwenForwardOutput, resize_rgb
from data_module.utils import MAX_RESOLUTION


@dataclasses.dataclass
class QwenMaskFlowPreprocessOutput(PreprocessOutput):

    raw_source: torch.Tensor | None = None
    mask: torch.Tensor | None = None


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

    # ---------------- Mask Operations ---------------- #
    def dilate_mask(self, mask: torch.Tensor, ks: int | None = None) -> torch.Tensor:
        return maskflow_utils.dilate_mask(mask, ks or self.mask_dilation_kernel)

    def blur_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return maskflow_utils.blur_mask(mask, self.mask_blur_kernel, self.mask_blur_sigma)

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

    # ---------------- Poisson Operations ---------------- #
    def neighbor_sum(self, x: torch.Tensor) -> torch.Tensor:
        return maskflow_utils.neighbor_sum(x)

    def neighbor_degree(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(x)
        return self.neighbor_sum(ones)

    def poisson_refine(self, g, x_S, M, soft_M, eps=1e-6, disable_progress_bar=False):
        return maskflow_utils.poisson_refine(
            g, x_S, M, soft_M, self.poisson_lambda_e, self.poisson_lambda_s,
            self.poisson_num_iter, self.poisson_momentum, eps, disable_progress_bar,
        )

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
                f"QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().preprocess_inputs(batch)

        source: torch.Tensor = conditions["source"].to(self.device, dtype=self.dtype)
        mask: torch.Tensor = conditions["mask"].to(self.device, dtype=self.dtype)

        # Resize all spatial inputs first.
        # masks use nearest-neighbor so their boundaries stay discrete.
        h, w = (target if target is not None else source).shape[-2:]
        aspect = w / h
        w, h = calculate_dimensions(MAX_RESOLUTION, aspect)
        if target is not None:
            target = resize_rgb(target, h, w)
        source = resize_rgb(source, h, w)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")

        # Apply morphology at model resolution so kernel sizes do not depend on
        # the uploaded image resolution.
        if self.mask_dilation_kernel > 0:
            mask = self.dilate_mask(mask, self.mask_dilation_kernel)
        if self.mask_blur_kernel > 0:
            mask = self.blur_mask(mask)

        if target is not None:
            target = self.image_processor.preprocess(target, h, w).unsqueeze(2)

        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, aspect)
        vlm_conditions = {
            "source": resize_rgb(source, ch, cw),
            "mask": F.interpolate(
                mask.float(),
                size=(ch, cw),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(dtype=mask.dtype),
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
            height=h, width=w,
            raw_source=raw_source,
            mask=mask,
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
                f"QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().prepare_forward_inputs(preprocessed_data)

        # ---------------- Encode Prompts ---------------- #
        prompt = preprocessed_data.prompt
        vlm_conditions = preprocessed_data.vlm_conditions
        dit_conditions = preprocessed_data.dit_conditions

        # TODO: There must be a more elegant way to implement CFG
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, vlm_conditions)

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
            tgt = self.poisson_refine(
                g=tgt.squeeze(2),
                x_S=conds[0].squeeze(2),
                M=mask_latents.squeeze(2) > 0,  # 1 for masked regions to be edited
                soft_M=mask_latents.squeeze(2),
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
        noise = torch.randn_like(tgt, generator=self.generator)
        ts = self.scheduler.sample_timesteps(tgt.shape[0], self.generator, self.device)
        sigmas = self.scheduler.get_sigmas(ts, img_seq_len=tgt.shape[1])

        xt = self.scheduler.add_noise_by_sigmas(noise, tgt, sigmas, source, mask_latents)
        gt = self.scheduler.get_velocity(noise, tgt, source, mask_latents)

        return QwenMaskFlowForwardOutput(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            noise=noise,
            noised_target=xt,
            timesteps=sigmas,  # Qwen use sigma-based denoise
            sigmas=sigmas,
            ground_truth=gt,
            conditions=cond_latents,
            image_shapes=image_shapes,
            mask_latents=mask_latents,
            mask_ratio=mask_ratio,
        )

    def prepare_eval_inputs(
        self,
        preprocessed_data: QwenMaskFlowPreprocessOutput,
        text_cfg_scale: float = 1.0,
    ) -> QwenMaskFlowForwardOutput:
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`; source supplies size without a target.
        """
        sample_mode = "argmax"

        if getattr(preprocessed_data, "mask", None) is None:
            logger.warning(
                f"QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().prepare_eval_inputs(preprocessed_data, text_cfg_scale)

        prompt = preprocessed_data.prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, preprocessed_data.vlm_conditions)

        text_cfg_enabled = text_cfg_scale > 1.0
        neg_prompt_embeds, neg_prompt_embeds_mask = None, None
        if text_cfg_enabled:
            neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt(
                preprocessed_data.negative_prompt, preprocessed_data.vlm_conditions
            )

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
        noise = torch.randn(noise_shape, generator=self.generator, device=self.device, dtype=self.dtype)
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
            # Positive prompt and mask
            "pm": QwenMaskFlowCFGBranch(
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                conditions=cond_latents,
                image_shapes=image_shapes,
            )
        }
        if text_cfg_enabled:
            # Negative prompt and mask
            cfg_branches["nm"] = QwenMaskFlowCFGBranch(
                prompt_embeds=neg_prompt_embeds,
                prompt_embeds_mask=neg_prompt_embeds_mask,
                conditions=cond_latents,
                image_shapes=image_shapes,
            )
        return QwenMaskFlowForwardOutput(
            height=height,
            width=width,
            noise=noise,
            conditions=cond_latents,
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
            edge_latents  (Tensor): The reshaped/VAE encoded softened binary edges.

        Return
            dict[str, Tensor]: The dict of different types losses.
        """
        loss_dict = {}
        loss_field = F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
        loss = None

        if self.enable_masked_loss and mask_latents is not None:
            loss_field = self.mask_loss_weight * mask_latents * loss_field

            if mask_ratio is not None:
                loss_field = (1.0 / (mask_ratio + 1e-6)) * loss_field
                loss_dict["mask_ratio"] = mask_ratio.mean()

        loss = (loss_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
        loss_dict["loss"] = loss
        return loss_dict

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        # `preprocess_inputs` includes extracting prompts and images from batched samples
        # and resize or conduct other augmentation on them.
        preprocessed_data = self.preprocess_inputs(batch)

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
            mask_latents=model_inputs.mask_latents,
        )
        return loss

    def combine_cfg_predictions(
        self,
        branch_predictions: dict[str, torch.Tensor],
        text_cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        cfg_outputs = branch_predictions["pm"]
        pos_norm = torch.norm(branch_predictions["pm"], dim=-1, keepdim=True).clamp(1e-6)

        # Negative prompt with mask
        if "nm" in branch_predictions:
            cfg_outputs = branch_predictions["nm"] + text_cfg_scale * (cfg_outputs - branch_predictions["nm"])

        cfg_norm = torch.norm(cfg_outputs, dim=-1, keepdim=True).clamp(1e-6)
        return (pos_norm / cfg_norm) * cfg_outputs

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
        x0_refined = self.poisson_refine(
            g=x0_pred,
            x_S=x_S,
            M=soft_M > 0,
            soft_M=soft_M,
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
        batch,
        num_inference_steps: int = 50,
        text_cfg_scale: float = 1.0,
    ) -> list[torch.Tensor]:
        r"""
        Evaluation step is used for batched sample evaluation, especially on benchmarks or testsets.
        """
        # As the batch share the same structure with training batch, we use the same extraction method as train.
        preprocessed_data = self.preprocess_inputs(batch)
        model_inputs = self.prepare_eval_inputs(preprocessed_data, text_cfg_scale)  # , mask_cfg_scale)
        xt = model_inputs.noise
        mask = preprocessed_data.mask
        raw_source = preprocessed_data.raw_source
        source = model_inputs.conditions[0]
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
                pred = self.combine_cfg_predictions(branch_preds, text_cfg_scale)

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
        prompt: str = None,
        negative_prompt: str = None,
        source_image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
        height: int | None = None,
        width: int | None = None,
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
    ) -> torch.Tensor:
        pass
