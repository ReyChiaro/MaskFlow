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

from tqdm import tqdm
from typing import Any, Literal, Optional
from loguru import logger

from schedulers import MaskFlowScheduler
from pipelines.base_pipeline import PreprocessOutput
from pipelines.qwenimage.qwenimage_edit_plus import QwenImageEditPlus, QwenForwardOutput
from data_module.utils import MAX_RESOLUTION


@dataclasses.dataclass
class QwenMaskFlowPreprocessOutput(PreprocessOutput):

    raw_source: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    edge: torch.Tensor | None = None
    mask_cfg_dropped: bool = False


@dataclasses.dataclass
class QwenMaskFlowCFGBranch:

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    conditions: list[torch.Tensor]
    image_shapes: list


@dataclasses.dataclass
class QwenMaskFlowForwardOutput(QwenForwardOutput):

    loss_weights: torch.Tensor | None = None

    # Mask and edge latents that are ``encoded'' by interpolation rather than VAE.
    mask_latents: torch.Tensor | None = None
    edge_latents: torch.Tensor | None = None

    mask_ratio: torch.Tensor | None = None

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
    mask_edge_width: int = 50

    mask_loss_weight: float = 1.0
    edge_loss_weight: float = 0.0

    # Always true
    enable_vae_mask_encoding: bool = True

    cfg_type: Literal["progressive", "condition_weighted"] = "condition_weighted"
    mask_cfg_null_type: Literal["full_one", "null"] = "null"
    enable_mask_cfg_gating: bool = False

    enable_masked_loss: bool = True

    enable_local_denoise_train: bool = False
    enable_local_denoise_infer: bool = False
    local_denoise_steps: list[float] = dataclasses.field(default_factory=list)

    enable_pixel_blend: bool = True

    enable_poisson_train: bool = True
    enable_poisson_infer: bool = True
    poisson_steps: list[float] = dataclasses.field(default_factory=list)
    poisson_lambda_e: float = 0.1
    poisson_lambda_s: float = 0.1
    poisson_num_iter: int = 50
    poisson_momentum: float = 0.1

    def __post_init__(self):
        if self.cfg_type not in {"progressive", "condition_weighted"}:
            raise ValueError(f"Unsupported cfg_type: {self.cfg_type}.")
        super().__post_init__()

    # ---------------- Mask Operations ---------------- #
    def dilate_mask(self, mask: torch.Tensor, ks: int | None = None) -> torch.Tensor:
        ks = ks or self.mask_dilation_kernel
        padding = ks // 2
        dilated = F.max_pool2d(mask, kernel_size=ks, padding=padding, stride=1)
        return dilated

    def erode_mask(self, mask: torch.Tensor, ks: int | None = None) -> torch.Tensor:
        ks = ks or self.mask_dilation_kernel
        mask = 1 - mask
        padding = ks // 2
        eroded = F.max_pool2d(mask, kernel_size=ks, padding=padding, stride=1)
        return 1.0 - eroded

    def blur_mask(self, mask: torch.Tensor) -> torch.Tensor:
        blur_mask_tensor = T.gaussian_blur(mask, kernel_size=self.mask_blur_kernel, sigma=self.mask_blur_sigma)
        blur_mask_tensor[mask < 1] = blur_mask_tensor[mask < 1] * 2
        blur_mask_tensor[mask >= 1] = 1
        return blur_mask_tensor

    def get_mask_edge(self, mask: torch.Tensor) -> torch.Tensor:
        return self.dilate_mask(mask, self.mask_edge_width // 2) - self.erode_mask(mask, self.mask_edge_width // 2)

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

    def prepare_null_mask_conditions(
        self,
        vlm_conditions: dict[str, torch.Tensor],
        dit_conditions: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.mask_cfg_null_type == "full_one":
            return (
                {
                    "source": vlm_conditions["source"],
                    "mask": torch.ones_like(vlm_conditions["mask"]),
                },
                {
                    "source": dit_conditions["source"],
                    "mask": torch.ones_like(dit_conditions["mask"]),
                },
            )
        if self.mask_cfg_null_type == "null":
            return {"source": vlm_conditions["source"]}, {"source": dit_conditions["source"]}
        raise ValueError(f"Unsupported mask_cfg_null_type: {self.mask_cfg_null_type}.")

    # ---------------- Poisson Operations ---------------- #
    def neighbor_sum(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kernel = x.new_tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ).view(1, 1, 3, 3)
        kernel = kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=C)

    def neighbor_degree(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(x)
        return self.neighbor_sum(ones)

    def poisson_refine(
        self,
        g: torch.Tensor,
        x_S: torch.Tensor,
        M: torch.Tensor,
        soft_M: torch.Tensor,
        eps: float = 1e-6,
        disable_progress_bar: bool = False,
    ):
        r"""
        Args
            g   (Tensor): The guidance tensor
            x_S (Tensor): The source tensor to keep background
            M   (Tensor): Mask
            soft_M (Optional Tensor): Soften mask for blending
        """
        B, C, H, W = g.shape
        M = M.float().expand(B, C, H, W)
        soft_M = soft_M.float().expand(B, C, H, W)

        # Initialization
        y = M * g + (1.0 - M) * x_S

        D = self.neighbor_degree(g)
        nsum_soft_M = self.neighbor_sum(soft_M)

        weight_sum = 0.5 * soft_M * D + 0.5 * nsum_soft_M

        nsum_g = self.neighbor_sum(g)
        nsum_g_soft_M = self.neighbor_sum(soft_M * g)
        weighted_nsum_g = 0.5 * soft_M * nsum_g + 0.5 * nsum_g_soft_M

        div_g = weight_sum * g - weighted_nsum_g

        x_S_out = (1.0 - M) * x_S
        nsum_x_S = self.neighbor_sum(x_S_out)
        nsum_x_S_soft_M = self.neighbor_sum(soft_M * x_S_out)
        soft_boundry = 0.5 * soft_M * nsum_x_S + 0.5 * nsum_x_S_soft_M

        diag = weight_sum + self.poisson_lambda_e * soft_M + self.poisson_lambda_s * (1.0 - soft_M)

        b = div_g + soft_boundry + self.poisson_lambda_e * soft_M * g + self.poisson_lambda_s * (1.0 - soft_M) * x_S

        # Solve linear
        for _ in tqdm(
            range(self.poisson_num_iter),
            desc="Solve Poisson",
            disable=disable_progress_bar,
        ):
            y_in = M * y
            nsum_y_in = self.neighbor_sum(y_in)
            nsum_y_in_soft_M = self.neighbor_sum(soft_M * y_in)

            y_next = (0.5 * soft_M * nsum_y_in + 0.5 * nsum_y_in_soft_M + b) / diag.clamp(min=eps)

            # Update masked area only
            y_next = M * y_next + (1.0 - M) * x_S

            y = self.poisson_momentum * y + (1.0 - self.poisson_momentum) * y_next

        if soft_M is not None:
            soft_M = soft_M.expand_as(g)
            y = soft_M * y + (1.0 - soft_M) * x_S
        return y

    def preprocess_inputs(self, batch: dict[str, Any]) -> QwenMaskFlowPreprocessOutput:
        r"""
        A batched data is supposed to have keys `prompt`, `target`.
        """
        prompt: list[str] = batch["prompt"]
        target: torch.Tensor = batch["target"].to(self.device, dtype=self.dtype)

        negative_prompt: Optional[list[str]] = batch.get("negative_prompt", None)
        conditions: Optional[dict[str, torch.Tensor]] = batch.get("conditions", None)

        if conditions is None or "mask" not in conditions or "source" not in conditions:
            logger.warning(
                f"QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().preprocess_inputs(batch)

        source: torch.Tensor = conditions["source"].to(self.device, dtype=self.dtype)
        mask: torch.Tensor = conditions["mask"].to(self.device, dtype=self.dtype)

        # ---------------- Preprocess ---------------- #
        # Preprocess mask and edge
        edge = self.get_mask_edge(mask)
        if self.mask_dilation_kernel > 0:
            mask = self.dilate_mask(mask, self.mask_dilation_kernel)
        if self.mask_blur_kernel > 0:
            mask = self.blur_mask(mask)
            edge = self.blur_mask(edge)

        # To tensor and reshape all images to target area as
        # the mask/source images are supposed to be same shapes.
        h, w = target.shape[-2:]
        aspect = w / h
        w, h = calculate_dimensions(MAX_RESOLUTION, aspect)
        target = self.image_processor.preprocess(target, h, w).unsqueeze(2)

        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, aspect)
        vlm_conditions = {
            "source": self.image_processor.resize(source, ch, cw),
            "mask": self.image_processor.resize(mask, ch, cw),
        }
        dit_conditions = {
            "source": self.image_processor.preprocess(source, h, w).unsqueeze(2),
            "mask": self.image_processor.preprocess(mask, h, w).unsqueeze(2),
        }
        raw_source = T.resize(source, [h, w], T.InterpolationMode.BILINEAR)
        mask = T.resize(mask, [h, w], T.InterpolationMode.NEAREST)
        edge = T.resize(edge, [h, w], T.InterpolationMode.NEAREST)

        return QwenMaskFlowPreprocessOutput(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vlm_conditions=vlm_conditions,
            dit_conditions=dit_conditions,
            target=target,
            raw_source=raw_source,
            mask=mask,
            edge=edge,
            mask_cfg_dropped=bool(batch.get("mask_cfg_dropped", False)),
        )

    def prepare_forward_inputs(self, preprocessed_data: QwenMaskFlowPreprocessOutput) -> QwenMaskFlowForwardOutput:
        r"""
        Prepare training forward inputs.
        The sample mode for VAE is fixed to `sample`, `target` must be provided.
        """
        sample_mode = "sample"

        if getattr(preprocessed_data, "mask", None) is None:
            logger.warning(
                f"QwenImageMaskFlow is used, but no mask and source found in dataset. Fall back to QwenImageEditPlus."
            )
            return super().prepare_forward_inputs(preprocessed_data)

        prompt = preprocessed_data.prompt
        vlm_conditions = preprocessed_data.vlm_conditions
        dit_conditions = preprocessed_data.dit_conditions
        if preprocessed_data.mask_cfg_dropped:
            vlm_conditions, dit_conditions = self.prepare_null_mask_conditions(vlm_conditions, dit_conditions)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, vlm_conditions)

        image_shapes = []
        target = preprocessed_data.target
        mask = preprocessed_data.mask
        edge = preprocessed_data.edge
        height, width = target.shape[-2:]

        mask_ratio = mask.flatten(1).sum(dim=-1, keepdim=True) / math.prod(mask.shape[1:])
        mask_ratio = mask_ratio.view(mask.shape[0], 1, 1)

        # ---------------- Encode and Pack ---------------- #
        # Encode target
        tgt = self.encode_image(target, sample_mode)

        # Encode condtions
        conds = [self.encode_image(dit_conditions["source"], sample_mode)]

        mask_latents = self.encode_mask(mask)
        edge_latents = self.encode_mask(edge)

        if "mask" in dit_conditions:
            conds.append(self.encode_image(dit_conditions["mask"], sample_mode))

        if self.enable_poisson_train:
            tgt_dtype = tgt.dtype
            tgt = (
                self.poisson_refine(
                    tgt.squeeze(2),
                    conds[0].squeeze(2),
                    mask_latents.squeeze(2) >= 1.0,
                    soft_M=mask_latents.squeeze(2),
                    disable_progress_bar=True,
                )
                .unsqueeze(2)
                .to(dtype=tgt_dtype)
            )

        image_shapes.append((1, tgt.shape[-2] // self.pacth_size, tgt.shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * tgt.shape[0]

        # Pack to 3D
        tgt = QwenImageEditPlusPipeline._pack_latents(tgt, tgt.shape[0], tgt.shape[1], tgt.shape[-2], tgt.shape[-1])
        cond_latents = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]

        # Pack mask and edge to 3D to satisfy the shapes of target and conditions.
        mask_latents = QwenImageEditPlusPipeline._pack_latents(
            mask_latents, mask_latents.shape[0], mask_latents.shape[1], mask_latents.shape[-2], mask_latents.shape[-1]
        )
        edge_latents = QwenImageEditPlusPipeline._pack_latents(
            edge_latents, edge_latents.shape[0], edge_latents.shape[1], edge_latents.shape[-2], edge_latents.shape[-1]
        )

        # --------------- Sample and Add Noise -------------- #
        source = cond_latents[0]
        noise = torch.randn_like(tgt, generator=self.generator)
        ts = self.scheduler.sample_timesteps(tgt.shape[0], self.generator, self.device)
        sigmas, loss_weights = self.scheduler.get_sigmas(ts, img_seq_len=tgt.shape[1], return_d_sigmas_dt=True)

        if self.enable_local_denoise_train:
            # Apply masks to vector fields prediction, only the masked area will be added noise
            # If MaskFlow scheduler is applied, then the area outside of the mask will be replaced
            # with a deterministic item.

            # The disabled samples' masks will be replaced by full-one (edit all) masks,
            # thus the full images will be added noises.
            disable_mask_ids = (sigmas < self.local_denoise_steps[0]) | (self.local_denoise_steps[1] < sigmas)

            # Assign all-one masks to original mask.
            # NOTE: This will affect the mask_latents in future use (e.g. loss calculation).
            mask_latents[disable_mask_ids, ...] = 1.0
            mask_ratio[disable_mask_ids, ...] = 1.0
        xt = self.scheduler.add_noise_by_sigmas(noise, tgt, sigmas, source, mask_latents)
        gt = self.scheduler.get_velocity(noise, tgt, source, mask_latents)
        while loss_weights.ndim < xt.ndim:
            loss_weights = loss_weights.unsqueeze(-1)

        return QwenMaskFlowForwardOutput(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            noise=noise,
            noised_target=xt,
            timesteps=sigmas,
            sigmas=sigmas,
            loss_weights=loss_weights,
            ground_truth=gt,
            conditions=cond_latents,
            image_shapes=image_shapes,
            mask_latents=mask_latents,
            edge_latents=edge_latents,
            mask_ratio=mask_ratio,
        )

    def prepare_eval_inputs(
        self,
        preprocessed_data: QwenMaskFlowPreprocessOutput,
        text_cfg_scale: float = 1.0,
        mask_cfg_scale: float = 1.0,
    ) -> QwenMaskFlowForwardOutput:
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`, `target` must be provided.
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
        mask_cfg_enabled = mask_cfg_scale > 1.0
        need_nm_branch = text_cfg_enabled or (self.cfg_type == "progressive" and mask_cfg_enabled)
        need_pn_branch = self.cfg_type == "condition_weighted" and mask_cfg_enabled
        need_nn_branch = self.cfg_type == "progressive" and mask_cfg_enabled

        neg_prompt_embeds, neg_prompt_embeds_mask = None, None
        pos_null_mask_prompt_embeds, pos_null_mask_prompt_embeds_mask = None, None
        neg_null_mask_prompt_embeds, neg_null_mask_prompt_embeds_mask = None, None
        null_dit_conditions = None
        if need_nm_branch or need_nn_branch:
            negative_prompt = preprocessed_data.negative_prompt
            neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt(
                negative_prompt, preprocessed_data.vlm_conditions
            )

        if mask_cfg_enabled:
            null_vlm_conditions, null_dit_conditions = self.prepare_null_mask_conditions(
                preprocessed_data.vlm_conditions,
                preprocessed_data.dit_conditions,
            )
            if need_pn_branch:
                pos_null_mask_prompt_embeds, pos_null_mask_prompt_embeds_mask = self.encode_prompt(
                    prompt, null_vlm_conditions
                )
            if need_nn_branch:
                neg_null_mask_prompt_embeds, neg_null_mask_prompt_embeds_mask = self.encode_prompt(
                    preprocessed_data.negative_prompt, null_vlm_conditions
                )

        image_shapes = []
        dit_conditions = preprocessed_data.dit_conditions
        target = preprocessed_data.target
        mask = preprocessed_data.mask
        edge = preprocessed_data.edge
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

        # Encode condtions
        conds = [
            self.encode_image(dit_conditions["source"], sample_mode),
            self.encode_image(dit_conditions["mask"], sample_mode),
        ]

        mask_latents = self.encode_mask(mask)
        edge_latents = self.encode_mask(edge)

        null_conds = None
        if null_dit_conditions is not None:
            null_conds = [conds[0]]
            if "mask" in null_dit_conditions:
                null_conds.append(self.encode_image(null_dit_conditions["mask"], sample_mode))

        image_shapes.append((1, noise_shape[-2] // self.pacth_size, noise_shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * noise.shape[0]

        mask_null_image_shapes = None
        if null_conds is not None:
            mask_null_image_shapes = [(1, noise_shape[-2] // self.pacth_size, noise_shape[-1] // self.pacth_size)]
            mask_null_image_shapes.extend(
                [(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in null_conds]
            )
            mask_null_image_shapes = [mask_null_image_shapes] * noise.shape[0]

        # Pack to 3D
        noise = QwenImageEditPlusPipeline._pack_latents(
            noise, noise_shape[0], noise_shape[1], noise_shape[-2], noise_shape[-1]
        )
        cond_latents = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]
        mask_null_cond_latents = None
        if null_conds is not None:
            mask_null_cond_latents = [
                QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1])
                for c in null_conds
            ]
        mask_latents = QwenImageEditPlusPipeline._pack_latents(
            mask_latents, mask_latents.shape[0], mask_latents.shape[1], mask_latents.shape[-2], mask_latents.shape[-1]
        )
        edge_latents = QwenImageEditPlusPipeline._pack_latents(
            edge_latents, edge_latents.shape[0], edge_latents.shape[1], edge_latents.shape[-2], edge_latents.shape[-1]
        )

        cfg_branches = {
            "pm": QwenMaskFlowCFGBranch(
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                conditions=cond_latents,
                image_shapes=image_shapes,
            )
        }
        if need_nm_branch:
            cfg_branches["nm"] = QwenMaskFlowCFGBranch(
                prompt_embeds=neg_prompt_embeds,
                prompt_embeds_mask=neg_prompt_embeds_mask,
                conditions=cond_latents,
                image_shapes=image_shapes,
            )
        if need_pn_branch:
            cfg_branches["pn"] = QwenMaskFlowCFGBranch(
                prompt_embeds=pos_null_mask_prompt_embeds,
                prompt_embeds_mask=pos_null_mask_prompt_embeds_mask,
                conditions=mask_null_cond_latents,
                image_shapes=mask_null_image_shapes,
            )
        if need_nn_branch:
            cfg_branches["nn"] = QwenMaskFlowCFGBranch(
                prompt_embeds=neg_null_mask_prompt_embeds,
                prompt_embeds_mask=neg_null_mask_prompt_embeds_mask,
                conditions=mask_null_cond_latents,
                image_shapes=mask_null_image_shapes,
            )

        return QwenMaskFlowForwardOutput(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            noise=noise,
            conditions=cond_latents,
            image_shapes=image_shapes,
            mask_latents=mask_latents,
            edge_latents=edge_latents,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_prompt_embeds_mask=neg_prompt_embeds_mask,
            cfg_branches=cfg_branches,
        )

    def compute_loss(
        self,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        loss_weights: torch.Tensor | None = None,
        mask_ratio: torch.Tensor | None = None,
        mask_latents: torch.Tensor | None = None,
        edge_latents: torch.Tensor | None = None,
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
        if loss_weights is not None:
            # no use
            loss_dict["loss_weights"] = loss_weights.mean()

        loss_field = F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
        loss = None

        if self.enable_masked_loss:
            if mask_latents is not None:
                mask_field = mask_latents * loss_field
                if mask_ratio is not None and self.mask_loss_weight > 0:
                    mask_field = self.mask_loss_weight * (1.0 / (mask_ratio + 1e-6)) * mask_field
                    loss_dict["mask_ratio"] = mask_ratio.mean()

                mask_loss = (mask_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
                loss_dict["mask_loss"] = mask_loss
                loss = mask_loss if loss is None else loss + mask_loss

            if edge_latents is not None and self.edge_loss_weight > 0:
                edge_field = self.edge_loss_weight * edge_latents * loss_field
                edge_loss = (edge_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
                loss_dict["edge_loss"] = edge_loss
                loss = edge_loss if loss is None else loss + edge_loss

        if loss is None:
            loss = (loss_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()

        loss_dict["loss"] = loss
        return loss_dict

    def forward_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
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
        loss = self.compute_loss(
            predictions=predictions,
            ground_truths=model_inputs.ground_truth,
            loss_weights=model_inputs.loss_weights,
            mask_ratio=model_inputs.mask_ratio,
            mask_latents=model_inputs.mask_latents,
            edge_latents=model_inputs.edge_latents,
        )
        return loss

    def denoise_cfg_branch(
        self,
        branch: QwenMaskFlowCFGBranch,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        transformer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        hidden_states = torch.cat([xt] + branch.conditions, dim=1)
        transformer = self.transformer if transformer is None else transformer
        return self.denoise_with_transformer(
            transformer=transformer,
            hidden_states=hidden_states,
            timesteps=timestep,
            prompt_embeds=branch.prompt_embeds,
            prompt_embeds_mask=branch.prompt_embeds_mask,
            img_shapes=branch.image_shapes,
            img_seq_len=xt.shape[1],
        )

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
        source4d = QwenImageEditPlusPipeline._unpack_latents(
            source, height, width, self.vae_scale_factor
        ).squeeze(2)
        mask4d = QwenImageEditPlusPipeline._unpack_latents(
            mask_latents, height, width, self.vae_scale_factor
        ).squeeze(2)

        x0_pred = self.scheduler.predict_x0(xt, curr_sigma, d_sigma_dt, pred, source, mask_latents, noise)
        x0_dtype = x0_pred.dtype
        x0_pred = QwenImageEditPlusPipeline._unpack_latents(
            x0_pred, height, width, self.vae_scale_factor
        ).squeeze(2)
        x0_refined = self.poisson_refine(
            x0_pred,
            source4d,
            mask4d >= 1.0,
            soft_M=mask4d,
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

    @staticmethod
    def rescale_cfg_prediction(cfg_pred: torch.Tensor, pos_pred: torch.Tensor) -> torch.Tensor:
        pos_norm = torch.norm(pos_pred, dim=-1, keepdim=True).clamp_min(1e-6)
        cfg_norm = torch.norm(cfg_pred, dim=-1, keepdim=True).clamp_min(1e-6)
        return (pos_norm / cfg_norm) * cfg_pred

    def combine_cfg_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        text_cfg_scale: float,
        mask_cfg_scale: float,
        mask_latents: torch.Tensor,
    ) -> torch.Tensor:
        pos_pred = predictions["pm"]
        text_cfg_enabled = text_cfg_scale > 1.0
        mask_cfg_enabled = mask_cfg_scale > 1.0
        if not text_cfg_enabled and not mask_cfg_enabled:
            return pos_pred

        if self.cfg_type == "progressive":
            null_text_mask_pred = predictions["nm"]
            if mask_cfg_enabled:
                null_text_null_mask_pred = predictions["nn"]
                mask_residual = null_text_mask_pred - null_text_null_mask_pred
                if self.enable_mask_cfg_gating:
                    mask_residual = mask_latents * mask_residual
                text_residual = pos_pred - null_text_mask_pred
                cfg_pred = (
                    null_text_null_mask_pred
                    + mask_cfg_scale * mask_residual
                    + text_cfg_scale * text_residual
                )
            else:
                cfg_pred = null_text_mask_pred + text_cfg_scale * (pos_pred - null_text_mask_pred)
            return self.rescale_cfg_prediction(cfg_pred, pos_pred)

        text_cfg_pred = pos_pred
        if text_cfg_enabled:
            null_text_mask_pred = predictions["nm"]
            text_cfg_pred = pos_pred + (text_cfg_scale - 1.0) * (pos_pred - null_text_mask_pred)

        mask_cfg_pred = pos_pred
        if mask_cfg_enabled:
            pos_prompt_null_mask_pred = predictions["pn"]
            mask_residual = pos_pred - pos_prompt_null_mask_pred
            if self.enable_mask_cfg_gating:
                mask_residual = mask_latents * mask_residual
            mask_cfg_pred = pos_pred + (mask_cfg_scale - 1.0) * mask_residual

        text_cfg_pred = self.rescale_cfg_prediction(text_cfg_pred, pos_pred)
        mask_cfg_pred = self.rescale_cfg_prediction(mask_cfg_pred, pos_pred)
        return 0.5 * (text_cfg_pred + mask_cfg_pred)

    def inference_mask(self, timestep: torch.Tensor, mask_latents: torch.Tensor) -> torch.Tensor:
        if not self.enable_local_denoise_infer:
            return mask_latents

        runtime_mask = mask_latents.clone()
        disable_mask_ids = (timestep < self.local_denoise_steps[0]) | (
            self.local_denoise_steps[1] < timestep
        )
        runtime_mask[disable_mask_ids, ...] = 1.0
        return runtime_mask

    @torch.inference_mode()
    def eval_step(
        self,
        batch,
        num_inference_steps: int = 50,
        text_cfg_scale: float = 1.0,
        mask_cfg_scale: float = 1.0,
        progress_callback=None,
    ) -> list[torch.Tensor]:
        preprocessed_data = self.preprocess_inputs(batch)
        model_inputs = self.prepare_eval_inputs(preprocessed_data, text_cfg_scale, mask_cfg_scale)
        xt = model_inputs.noise
        mask = preprocessed_data.mask
        edge = preprocessed_data.edge
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
                text_cfg_enabled = text_cfg_scale > 1.0
                mask_cfg_enabled = mask_cfg_scale > 1.0

                predictions = {"pm": self.denoise_cfg_branch(cfg_branches["pm"], xt, timestep)}
                if text_cfg_enabled or (self.cfg_type == "progressive" and mask_cfg_enabled):
                    predictions["nm"] = self.denoise_cfg_branch(cfg_branches["nm"], xt, timestep)
                if mask_cfg_enabled:
                    null_mask_branch = "nn" if self.cfg_type == "progressive" else "pn"
                    predictions[null_mask_branch] = self.denoise_cfg_branch(
                        cfg_branches[null_mask_branch], xt, timestep
                    )

                pred = self.combine_cfg_predictions(
                    predictions,
                    text_cfg_scale,
                    mask_cfg_scale,
                    mask_latents,
                )

                if self.enable_poisson_infer and self.poisson_steps[0] <= t.item() <= self.poisson_steps[1]:
                    pred = self.apply_poisson_to_prediction(
                        xt,
                        pred,
                        curr_sigma,
                        d_sigma_dt,
                        source,
                        mask_latents,
                        noise,
                        model_inputs.height,
                        model_inputs.width,
                        disable_progress_bar=True,
                    )

                runtime_mask = self.inference_mask(timestep, mask_latents)
                xt = self.scheduler.step(
                    xt,
                    pred,
                    curr_sigma,
                    next_sigma,
                    d_sigma_dt,
                    source,
                    runtime_mask,
                    noise,
                )
                if progress_callback is not None:
                    progress_callback(step_index, num_inference_steps)

        output = QwenImageEditPlusPipeline._unpack_latents(
            xt, model_inputs.height, model_inputs.width, self.vae_scale_factor
        )
        output = self.decode_image(output)

        if self.enable_pixel_blend:
            output = mask * output + (1.0 - mask) * raw_source

        return {"mask": mask, "edge": edge, "output": output}
