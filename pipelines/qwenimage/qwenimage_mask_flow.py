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
from typing import Any, Optional
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


@dataclasses.dataclass
class QwenMaskFlowForwardOutput(QwenForwardOutput):

    # Mask and edge latents that are ``encoded'' by interpolation rather than VAE.
    mask_latents: torch.Tensor | None = None
    edge_latents: torch.Tensor | None = None

    mask_ratio: torch.Tensor | None = None


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

    enable_vae_mask_encoding: bool = True

    enable_masked_loss: bool = True
    enable_mask_denoise_train: bool = True
    enable_mask_denoise_infer: bool = True
    mask_denoise_steps: list[float] = dataclasses.field(default_factory=list)

    enable_pixel_blend: bool = True

    enable_poisson_train: bool = True
    enable_poisson_infer: bool = True
    poisson_steps: list[float] = dataclasses.field(default_factory=list)
    poisson_lambda_color: float = 0.1
    poisson_num_iter: int = 10
    poisson_momentum: float = 0.1

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
        soft_M: torch.Tensor | None = None,
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

        # Initialization
        y = M * g + (1.0 - M) * x_S
        D = self.neighbor_degree(g)
        nsum_g = self.neighbor_sum(g)
        div_g = D * g - nsum_g

        x_S_out = (1.0 - M) * x_S
        nsum_x_S = self.neighbor_sum(x_S_out)

        b = div_g + self.poisson_lambda_color * g + nsum_x_S

        diag = D + self.poisson_lambda_color

        # Solve linear
        for _ in tqdm(
            range(self.poisson_num_iter),
            desc="Solve Poisson",
            disable=disable_progress_bar,
        ):
            y_in = M * y
            nsum_y_in = self.neighbor_sum(y_in)

            y_next = (nsum_y_in + b) / diag.clamp(min=1e-6)

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

        # Conduct CFG dropout
        prompt = preprocessed_data.prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, preprocessed_data.vlm_conditions)

        image_shapes = []
        dit_conditions = preprocessed_data.dit_conditions
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

        if self.enable_vae_mask_encoding:
            # This decides whether the input mask is sparse.
            # If encoded with VAE, then the zero will (large probably) be mapped to a non-zero value.
            conds.append(self.encode_image(dit_conditions["mask"], sample_mode))
        else:
            conds.append(mask_latents.clone())

        if self.enable_poisson_train:
            tgt = self.poisson_refine(tgt, conds[0], mask_latents >= 1.0, mask_latents)

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
        sigmas = self.scheduler.get_sigmas(ts, img_seq_len=tgt.shape[1])

        if self.enable_mask_denoise_train:
            # Apply masks to vector fields prediction, only the masked area will be added noise
            # If MaskFlow scheduler is applied, then the area outside of the mask will be replaced
            # with a deterministic item.

            # The disabled samples' masks will be replaced by full-one (edit all) masks,
            # thus the full images will be added noises.
            disable_mask_ids = (sigmas < self.mask_denoise_steps[0]) | (self.mask_denoise_steps[1] < sigmas)

            # Assign all-one masks to original mask.
            # NOTE: This will affect the mask_latents in future use (e.g. loss calculation).
            mask_latents[disable_mask_ids, ...] = 1.0
            mask_ratio[disable_mask_ids, ...] = 1.0
        xt = self.scheduler.add_noise_by_sigmas(noise, tgt, sigmas, source, mask_latents)
        gt = self.scheduler.get_velocity(noise, tgt, source, mask_latents)

        return QwenMaskFlowForwardOutput(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            noise=noise,
            noised_target=xt,
            timesteps=sigmas,
            sigmas=sigmas,
            ground_truth=gt,
            conditions=cond_latents,
            image_shapes=image_shapes,
            mask_latents=mask_latents,
            edge_latents=edge_latents,
            mask_ratio=mask_ratio,
        )

    def prepare_eval_inputs(
        self, preprocessed_data: QwenMaskFlowPreprocessOutput, cfg_scale: float = 1.0
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
            return super().prepare_eval_inputs(preprocessed_data, cfg_scale)

        prompt = preprocessed_data.prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, preprocessed_data.vlm_conditions)

        neg_prompt_embeds, neg_prompt_embeds_mask = None, None
        if cfg_scale > 1.0:
            negative_prompt = preprocessed_data.negative_prompt
            neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt(
                negative_prompt, preprocessed_data.vlm_conditions
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
        conds = [self.encode_image(dit_conditions["source"], sample_mode)]

        mask_latents = self.encode_mask(mask)
        edge_latents = self.encode_mask(edge)

        if self.enable_vae_mask_encoding:
            # This decides whether the input mask is sparse.
            # If encoded with VAE, then the zero will (large probably) be mapped to a non-zero value.
            conds.append(self.encode_image(dit_conditions["mask"], sample_mode))
        else:
            conds.append(mask_latents.clone())

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
        edge_latents = QwenImageEditPlusPipeline._pack_latents(
            edge_latents, edge_latents.shape[0], edge_latents.shape[1], edge_latents.shape[-2], edge_latents.shape[-1]
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
        )

    def compute_loss(
        self,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
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
        loss_field = F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
        loss = None
        loss_dict = {}

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
            mask_ratio=model_inputs.mask_ratio,
            mask_latents=model_inputs.mask_latents,
            edge_latents=model_inputs.edge_latents,
        )
        return loss

    @torch.inference_mode()
    def eval_step(self, batch, num_inference_steps: int = 50, cfg_scale: float = 4.0) -> list[torch.Tensor]:
        preprocessed_data = self.preprocess_inputs(batch)
        model_inputs = self.prepare_eval_inputs(preprocessed_data, cfg_scale)
        xt = model_inputs.noise
        mask = preprocessed_data.mask
        edge = preprocessed_data.edge
        raw_source = preprocessed_data.raw_source
        source = model_inputs.conditions[0]
        mask_latents = model_inputs.mask_latents
        noise = copy.deepcopy(model_inputs.noise)
        source4d = QwenImageEditPlusPipeline._unpack_latents(
            source, model_inputs.height, model_inputs.width, self.vae_scale_factor
        ).squeeze(2)
        mask4d = QwenImageEditPlusPipeline._unpack_latents(
            mask_latents, model_inputs.height, model_inputs.width, self.vae_scale_factor
        ).squeeze(2)

        with self.scheduler.inference(num_inference_steps, img_seq_len=xt.shape[1]) as inferencer:
            for t, curr_sigma, next_sigma in tqdm(inferencer, total=num_inference_steps):
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
                if cfg_scale > 1.0 and model_inputs.negative_prompt_embeds is not None:
                    neg_pred = self.denoise(
                        hidden_states=hidden_states,
                        timesteps=timestep,
                        prompt_embeds=model_inputs.negative_prompt_embeds,
                        prompt_embeds_mask=model_inputs.negative_prompt_embeds_mask,
                        img_shapes=model_inputs.image_shapes,
                        img_seq_len=xt.shape[1],
                    )
                    cfg_pred = neg_pred + cfg_scale * (pred - neg_pred)
                    pred_norm = torch.norm(pred, dim=-1, keepdim=True)
                    cfg_norm = torch.norm(cfg_pred, dim=-1, keepdim=True)
                    pred = (pred_norm / cfg_norm) * cfg_pred

                if self.enable_mask_denoise_infer:
                    # For inference, the mask will be changed in-place,
                    # so we must clone it for every timestep.
                    runtime_mask = mask_latents.clone()
                    disable_mask_ids = (timestep < self.mask_denoise_steps[0]) | (self.mask_denoise_steps[1] < timestep)
                    runtime_mask[disable_mask_ids, ...] = 1.0
                    xt = self.scheduler.step(xt, pred, curr_sigma, next_sigma, source, runtime_mask, noise)
                else:
                    xt = self.scheduler.step(xt, pred, curr_sigma, next_sigma, source, mask_latents, noise)

                if self.enable_poisson_infer and self.poisson_steps[0] <= t.item() < self.poisson_steps[1]:
                    xt = QwenImageEditPlusPipeline._unpack_latents(
                        xt, model_inputs.height, model_inputs.width, self.vae_scale_factor
                    ).squeeze(2)

                    xt = self.poisson_refine(xt, source4d, mask4d >= 1.0, soft_M=mask4d)
                    xt = QwenImageEditPlusPipeline._pack_latents(
                        xt, xt.shape[0], xt.shape[1], xt.shape[-2], xt.shape[-1]
                    )

        output = QwenImageEditPlusPipeline._unpack_latents(
            xt, model_inputs.height, model_inputs.width, self.vae_scale_factor
        )
        output = self.decode_image(output)

        if self.enable_pixel_blend:
            output = mask * output + (1.0 - mask) * raw_source

        return {"mask": mask, "edge": edge, "output": output}
