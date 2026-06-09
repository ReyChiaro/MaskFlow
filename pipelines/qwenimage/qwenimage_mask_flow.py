import copy
import math
import random
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
from typing import Any, Iterable
from loguru import logger

from schedulers import MaskFlowScheduler
from .qwenimage_edit_plus import QwenImageEditPlus
from data_module.utils import (
    reshape_to_divisible_max_resolution,
    crop_image_to_aspect_ratio,
    ASPECT_RATIOS,
    MAX_RESOLUTION,
    MAX_CONDITION_RESOLUTION,
    DIVISIBLE_BY,
)


@dataclasses.dataclass
class QwenImageMaskFlow(QwenImageEditPlus):

    scheduler: MaskFlowScheduler | None = None

    mask_dilation_kernel: int = 45
    mask_blur_kernel: int = 45
    mask_blur_sigma: float = 25.0
    mask_edge_width: int = 90  # dilate 45, erode 45

    mask_loss_weight: float = 1.0
    edge_loss_weight: float = 0.0

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

    # -------------------------------------------------- #

    def preprocess_inputs(self, batch) -> dict[str, Any]:
        r"""
        A batched data is supposed to have keys `prompt`, `conditions` and `target`.
        """
        prompt: str | list[str] = batch["prompt"]
        negative_prompt: str | list[str] = batch.get("negative_prompt", None)
        source = batch["conditions"][0].to(self.device, dtype=self.dtype)
        mask = batch["conditions"][1].to(self.device, dtype=self.dtype)
        target: torch.Tensor = batch["target"].to(self.device, dtype=self.dtype)

        # ---------------- Preprocess ---------------- #
        # Preprocess mask and edge
        edge = self.dilate_mask(mask, self.mask_edge_width // 2) - self.erode_mask(mask, self.mask_edge_width // 2)
        if self.mask_dilation_kernel > 0:
            mask = self.dilate_mask(mask, self.mask_dilation_kernel)
        if self.mask_blur_kernel > 0:
            mask = self.blur_mask(mask)
            edge = self.blur_mask(edge)

        conditions: list[torch.Tensor] = [source, mask]
        mask_ratio = mask.sum(dim=(-2, -1, 1), keepdim=True) / (mask.shape[-2] * mask.shape[-1] * mask.shape[1])
        mask_ratio = mask_ratio.view(mask.shape[0], 1, 1)

        mask_image = mask.clone()
        edge_image = edge.clone()
        raw_target = target.clone()

        # To tensor and reshape to target area
        h, w = target.shape[-2:]
        aspect = w / h
        cond_h, cond_w = calculate_dimensions(CONDITION_IMAGE_SIZE, aspect)
        conditions_vlm = [self.image_processor.resize(c, cond_h, cond_w) for c in conditions]
        target = self.image_processor.preprocess(target, h, w).unsqueeze(2)
        conditions_dit = [self.image_processor.preprocess(c, h, w).unsqueeze(2) for c in conditions]
        mask_image = T.resize(mask_image, [h, w], T.InterpolationMode.NEAREST)
        edge_image = T.resize(edge_image, [h, w], T.InterpolationMode.NEAREST)

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "conditions_vlm": conditions_vlm,
            "conditions_dit": conditions_dit,
            "target": target,
            "mask_image": mask_image,
            "edge_image": edge_image,
            "mask_ratio": mask_ratio,
            "raw_target": raw_target,
        }

    def prepare_forward_inputs(self, batch):
        r"""
        Prepare training forward inputs.
        The sample mode for VAE is fixed to `sample`, `target` must be provided.
        """
        sample_mode = "sample"
        processed_data = self.preprocess_inputs(batch)

        # Conduct CFG dropout
        prompt = processed_data["prompt"]
        if random.random() < self.cfg_dropout:
            prompt = ""
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, processed_data["conditions_vlm"])

        image_shapes = []
        conditions_dit = processed_data["conditions_dit"]
        target = processed_data["target"]
        mask_image = processed_data["mask_image"]
        edge_image = processed_data["edge_image"]

        # ---------------- Encode and Pack ---------------- #
        # Encode
        x0 = self.encode_image(target, sample_mode)
        mask_latents = self.encode_mask(mask_image)
        edge_latents = self.encode_mask(edge_image)
        conds = [self.encode_image(c, sample_mode) for c in conditions_dit]
        image_shapes.append((1, x0.shape[-2] // self.pacth_size, x0.shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * x0.shape[0]

        # Pack to 3D
        x0 = QwenImageEditPlusPipeline._pack_latents(x0, x0.shape[0], x0.shape[1], x0.shape[-2], x0.shape[-1])
        cond_latents = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]
        mask_latents = QwenImageEditPlusPipeline._pack_latents(
            mask_latents, mask_latents.shape[0], mask_latents.shape[1], mask_latents.shape[-2], mask_latents.shape[-1]
        )
        edge_latents = QwenImageEditPlusPipeline._pack_latents(
            edge_latents, edge_latents.shape[0], edge_latents.shape[1], edge_latents.shape[-2], edge_latents.shape[-1]
        )
        source = cond_latents[0]

        # --------------- Sample and Add Noise -------------- #
        noise = torch.randn_like(x0, generator=self.generator)
        ts = self.scheduler.sample_timesteps(x0.shape[0], self.generator, self.device)
        xt, sigmas = self.scheduler.add_noise(noise, x0, ts, source, mask_latents)
        gt = self.scheduler.get_velocity(noise, x0, source, mask_latents)

        return {
            "height": target.shape[-2],
            "width": target.shape[-1],
            "timesteps": ts,
            "sigmas": sigmas,
            "noise": noise,
            "gt": gt,
            "mask_image": mask_image,
            "edge_image": edge_image,
            "mask_latents": mask_latents,
            "edge_latents": edge_latents,
            "mask_ratio": processed_data["mask_ratio"],
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "xt": xt,
            "x0": x0,
            "conditions": cond_latents,
            "image_shapes": image_shapes,
        }

    def prepare_eval_inputs(self, batch, cfg_scale: float = 0):
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`, `target` must be provided.
        """
        sample_mode = "argmax"
        processed_data = self.preprocess_inputs(batch)

        prompt = processed_data["prompt"]
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, processed_data["conditions_vlm"])

        neg_prompt_embeds, neg_prompt_embeds_mask = None, None
        if cfg_scale > 0:
            negative_prompt = processed_data["negative_prompt"]
            neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt(
                negative_prompt, processed_data["conditions_vlm"]
            )

        image_shapes = []
        conditions_dit = processed_data["conditions_dit"]
        target = processed_data["target"]
        mask_image = processed_data["mask_image"]
        edge_image = processed_data["edge_image"]

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
        mask_latents = self.encode_mask(mask_image)
        edge_latents = self.encode_mask(edge_image)
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
        mask_latents = QwenImageEditPlusPipeline._pack_latents(
            mask_latents, mask_latents.shape[0], mask_latents.shape[1], mask_latents.shape[-2], mask_latents.shape[-1]
        )
        edge_latents = QwenImageEditPlusPipeline._pack_latents(
            edge_latents, edge_latents.shape[0], edge_latents.shape[1], edge_latents.shape[-2], edge_latents.shape[-1]
        )

        return {
            "height": target.shape[-2],
            "width": target.shape[-1],
            "noise": noise,
            "mask_latents": mask_latents,
            "edge_latents": edge_latents,
            "mask_image": mask_image,
            "edge_image": edge_image,
            "target": processed_data["raw_target"],
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": neg_prompt_embeds,
            "negative_prompt_embeds_mask": neg_prompt_embeds_mask,
            "conditions": cond_latents,
            "image_shapes": image_shapes,
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        mask_ratio: torch.Tensor | None = None,
        mask_latents: torch.Tensor | None = None,
        edge_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Compute loss with masks and edges.
        """
        loss_field = F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
        loss = None
        loss_dict = {}

        if mask_latents is not None:
            mask_field = mask_latents * loss_field
            if mask_ratio is not None and self.mask_loss_weight > 0:
                mask_field = self.mask_loss_weight * (1.0 / (mask_ratio + 1e-6)) * mask_field
                loss_dict["mask_ratio"] = mask_ratio.mean()

            mask_loss = (mask_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
            loss_dict["mask_loss"] = mask_loss
            loss = mask_loss if loss is None else loss + mask_loss

        if edge_latents is not None and self.edge_loss_weight > 0:
            # edge_field = self.edge_loss_weight * edge * loss_field
            # edge_loss = (edge_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
            # loss_dict["edge_loss"] = edge_loss
            # loss_dict["loss"] = loss_dict["loss"] + edge_loss
            pass

        if loss is None:
            loss = (loss_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
        loss_dict["loss"] = loss

        return loss_dict

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
        loss = self.compute_loss(
            predictions,
            inputs["gt"],
            mask_ratio=inputs["mask_ratio"],
            mask_latents=inputs["mask_latents"],
            edge_latents=inputs["edge_latents"],
        )
        return loss

    @torch.inference_mode()
    def eval_step(self, batch, global_step, num_inference_steps: int = 50, cfg_scale: float = 0) -> list:
        inputs = self.prepare_eval_inputs(batch, cfg_scale)
        xt = inputs["noise"]
        source = inputs["conditions"][0]
        mask = inputs["mask_latents"]
        mask_image = inputs["mask_image"]
        edge_image = inputs["edge_image"]

        step = 0
        with self.scheduler.inference_sampler(xt, num_inference_steps, source, mask, xt.shape[1]) as sampler:
            for xt, t, inferencer in tqdm(sampler, total=num_inference_steps):
                step += 1
                hidden_states = torch.cat([xt] + [c for c in inputs["conditions"]], dim=1)
                timestep = t.expand(hidden_states.shape[0]).to(device=self.device, dtype=self.dtype)

                pred = self.denoise(
                    hidden_states,
                    timestep,
                    inputs["prompt_embeds"],
                    inputs["prompt_embeds_mask"],
                    inputs["image_shapes"],
                    xt.shape[1],
                )

                # Do CFG
                if cfg_scale > 0:
                    neg_pred = self.denoise(
                        hidden_states,
                        timestep,
                        inputs["negative_prompt_embeds"],
                        inputs["negative_prompt_embeds_mask"],
                        inputs["image_shapes"],
                        xt.shape[1],
                    )
                    cfg_pred = neg_pred + cfg_scale * (pred - neg_pred)

                    pred_norm = torch.norm(pred, dim=-1, keepdim=True)
                    cfg_norm = torch.norm(cfg_pred, dim=-1, keepdim=True)
                    pred = (pred_norm / cfg_norm) * cfg_pred

                inferencer.step(pred)

        output = QwenImageEditPlusPipeline._unpack_latents(xt, inputs["height"], inputs["width"], self.vae_scale_factor)
        output = self.decode_image(output)
        return [mask_image, edge_image, output]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None = None,
        image: Image.Image | list[Image.Image] | None = None,
        negative_prompt: str | None = None,
        mask: Image.Image | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        cfg_scale: float = 0.0,
        **kwargs,
    ):
        r"""
        Args
            kwargs:
                - mask (default: None)
                - negative_prompt (default: None)
                - num_inference_steps (default: 50)
                - height (default: 1024)
                - width (default: 1024)
                - cfg_scale (default: 0)
        """
        prompt = "" if prompt is None else prompt

        # Target aspect ratio
        raw_ar = width / height
        tgt_ar = min(ASPECT_RATIOS, key=lambda x: abs(raw_ar - int(x.split(":")[0]) / int(x.split(":")[1])))
        tgt_ar = int(tgt_ar.split(":")[0]) / int(tgt_ar.split(":")[1])
        height = int(math.sqrt(MAX_RESOLUTION / tgt_ar))
        width = int(math.sqrt(MAX_RESOLUTION * tgt_ar))
        height = height // DIVISIBLE_BY * DIVISIBLE_BY
        width = width // DIVISIBLE_BY * DIVISIBLE_BY

        image_vlm = image_dit = None
        if image is not None:
            if not isinstance(image, Iterable):
                image = [image]

            if len(image) > 1 and mask is not None:
                logger.warning(f"Mask is provided but find multiple images are provided. \
                    If you want to use MaskFlow image edit, provide ONE image \
                    that you want to edit and the mask instead, otherwise the \
                    mask will be ignored.")

                image = [image[0]]

            if len(image) == 1 and mask is not None:
                # Use mask-based image editing
                if height != image[0].height or width != image[0].width:
                    logger.warning(
                        f"In Mask-Based image editing, the specific ({height=}, {width=}) is not equal to the image(source) shape ({image[0].height=}, {image[0].width}) or mask shape ({mask.height=}, {mask.width=})."
                    )
                    height = image[0].height
                    width = image[0].width

                    # Target aspect ratio
                    raw_ar = width / height
                    tgt_ar = min(ASPECT_RATIOS, key=lambda x: abs(raw_ar - int(x.split(":")[0]) / int(x.split(":")[1])))
                    tgt_ar = int(tgt_ar.split(":")[0]) / int(tgt_ar.split(":")[1])
                    height = int(math.sqrt(MAX_RESOLUTION / tgt_ar))
                    width = int(math.sqrt(MAX_RESOLUTION * tgt_ar))
                    height = height // DIVISIBLE_BY * DIVISIBLE_BY
                    width = width // DIVISIBLE_BY * DIVISIBLE_BY

            # Handle image and re-calculate image shapes
            image = [T.to_tensor(i).unsqueeze(0).to(self.device, dtype=self.dtype) for i in image]

            # Handle mask
            mask_image = None
            if mask is not None:
                mask = T.to_tensor(mask).unsqueeze(0).to(self.device, dtype=self.dtype)
                mask = T.resize(mask, [height, width], T.InterpolationMode.NEAREST)
                if self.mask_dilation_kernel > 0:
                    mask = self.dilate_mask(mask, self.mask_dilation_kernel)
                if self.mask_blur_kernel > 0:
                    mask = self.blur_mask(mask)
                mask_image = mask.clone()
                image.append(mask)

            # Condition aspect ratio
            if mask_image is not None:
                # Mask-based image editing should keep the source, mask and noise images shapes the same
                image = [T.resize(i, [height, width]) for i in image]
                image_aspect = [(i, width / height) for i in image]
            else:
                image_aspect = [crop_image_to_aspect_ratio(i) for i in image]
                image = [reshape_to_divisible_max_resolution(i, ar, MAX_RESOLUTION) for i, ar in image_aspect]
            image_vlm = [reshape_to_divisible_max_resolution(i, ar, MAX_CONDITION_RESOLUTION) for i, ar in image_aspect]
            image_dit = [self.image_processor.preprocess(c, c.shape[-2], c.shape[-1]).unsqueeze(2) for c in image]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt, image_vlm)
        neg_prompt_embeds, neg_prompt_embeds_mask = None, None
        if negative_prompt is not None and cfg_scale > 0:
            neg_prompt_embeds, neg_prompt_embeds_mask = self.encode_prompt(negative_prompt, image_vlm)

        image_shapes = []
        noise_shape = (1, self.vae_channels, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
        noise_latents = torch.randn(noise_shape, generator=self.generator, device=self.device, dtype=self.dtype)
        noise_latents = QwenImageEditPlusPipeline._pack_latents(
            noise_latents, noise_shape[0], noise_shape[1], noise_shape[-2], noise_shape[-1]
        )
        image_shapes.append((1, noise_shape[-2] // self.pacth_size, noise_shape[-1] // self.pacth_size))

        image_latents = mask_latents = None
        if image is not None:
            if mask_image is not None:
                mask_latents = self.encode_mask(mask)
                mask_latents = QwenImageEditPlusPipeline._pack_latents(
                    mask_latents,
                    mask_latents.shape[0],
                    mask_latents.shape[1],
                    mask_latents.shape[-2],
                    mask_latents.shape[-1],
                )

            image_latents = [self.encode_image(c, "argmax") for c in image_dit]
            image_shapes.extend(
                [(1, i.shape[-2] // self.pacth_size, i.shape[-1] // self.pacth_size) for i in image_latents]
            )
            image_latents = [
                QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1])
                for c in image_latents
            ]
            image_shapes = [image_shapes]

        # ---------------- Denoise ---------------- #
        xt = noise_latents
        source = None if image_latents is None else image_latents[0]
        with self.scheduler.inference_sampler(xt, num_inference_steps, source, mask_latents, xt.shape[1]) as sampler:
            for xt, t, inferencer in tqdm(sampler, total=num_inference_steps):
                hidden_states = xt
                if image_latents is not None:
                    hidden_states = torch.cat([hidden_states] + [c for c in image_latents], dim=1)

                timestep = t.expand(hidden_states.shape[0]).to(device=self.device, dtype=self.dtype)
                pred = self.denoise(
                    hidden_states, timestep, prompt_embeds, prompt_embeds_mask, image_shapes, xt.shape[1]
                )

                # Do CFG
                if cfg_scale > 0:
                    neg_pred = self.denoise(
                        hidden_states, timestep, neg_prompt_embeds, neg_prompt_embeds_mask, image_shapes, xt.shape[1]
                    )
                    cfg_pred = neg_pred + cfg_scale * (pred - neg_pred)
                    pred_norm = torch.norm(pred, dim=-1, keepdim=True)
                    cfg_norm = torch.norm(cfg_pred, dim=-1, keepdim=True)
                    pred = (pred_norm / cfg_norm) * cfg_pred

                inferencer.step(pred)

        output = QwenImageEditPlusPipeline._unpack_latents(xt, height, width, self.vae_scale_factor)
        output = self.decode_image(output)
        return T.to_pil_image(output[0].float())
