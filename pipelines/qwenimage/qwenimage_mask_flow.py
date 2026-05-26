import copy
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

from typing import Any

from schedulers import MaskFlowScheduler
from .qwenimage_edit_plus import QwenImageEditPlus


@dataclasses.dataclass
class QwenImageMaskFlow(QwenImageEditPlus):

    scheduler: MaskFlowScheduler

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
        raw_batch = copy.deepcopy(batch)
        prompt: str | list[str] = batch["prompt"]
        negative_prompt: str | list[str] = batch["negative_prompt"]
        conditions: list[torch.Tensor] = [c.to(self.device, dtype=self.dtype) for c in batch["conditions"]]
        target: torch.Tensor = batch["target"].to(self.device, dtype=self.dtype)

        # ---------------- Preprocess ---------------- #
        # Preprocess mask and edge
        mask = raw_batch["conditions"][1].to(self.device, dtype=self.dtype)
        if self.mask_dilation_kernel > 0:
            mask = self.dilate_mask(mask, self.mask_dilation_kernel)
        edge = self.dilate_mask(mask, self.mask_edge_width // 2) - self.erode_mask(mask, self.mask_edge_width // 2)
        if self.mask_blur_kernel > 0:
            mask = self.blur_mask(mask)
            edge = self.blur_mask(edge)

        # To tensor and reshape to target areas
        h, w = target.shape[-2:]
        aspect = w / h
        cond_h, cond_w = calculate_dimensions(CONDITION_IMAGE_SIZE, aspect)
        conditions_vlm = [self.image_processor.resize(c, cond_h, cond_w) for c in conditions]
        target = self.image_processor.preprocess(target, h, w).unsqueeze(2)
        conditions_dit = [self.image_processor.preprocess(c, h, w).unsqueeze(2) for c in conditions]
        mask = self.image_processor.resize(mask, h, w)
        edge = self.image_processor.resize(edge, h, w)

        return {
            "raw": raw_batch,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "mask": mask,
            "edge": edge,
            "conditions_vlm": conditions_vlm,
            "conditions_dit": conditions_dit,
            "target": target,
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
        mask = processed_data["mask"]
        edge = processed_data["edge"]

        # ---------------- Encode and Pack ---------------- #
        # Encode
        tgt = self.encode_image(target, sample_mode)
        mask = self.encode_mask(mask)
        edge = self.encode_mask(edge)
        conds = [self.encode_image(c, sample_mode) for c in conditions_dit]
        image_shapes.append((1, tgt.shape[-2] // self.pacth_size, tgt.shape[-1] // self.pacth_size))
        image_shapes.extend([(1, c.shape[-2] // self.pacth_size, c.shape[-1] // self.pacth_size) for c in conds])
        image_shapes = [image_shapes] * tgt.shape[0]

        # Pack to 3D
        x0 = QwenImageEditPlusPipeline._pack_latents(tgt, tgt.shape[0], tgt.shape[1], tgt.shape[-2], tgt.shape[-1])
        cond_latents = [
            QwenImageEditPlusPipeline._pack_latents(c, c.shape[0], c.shape[1], c.shape[-2], c.shape[-1]) for c in conds
        ]
        mask = QwenImageEditPlusPipeline._pack_latents(
            mask, mask.shape[0], mask.shape[1], mask.shape[-2], mask.shape[-1]
        )
        edge = QwenImageEditPlusPipeline._pack_latents(
            edge, edge.shape[0], edge.shape[1], edge.shape[-2], edge.shape[-1]
        )
        source = cond_latents[0]

        # --------------- Sample and Add Noise -------------- #
        noise = torch.randn_like(x0, generator=self.generator)
        ts = self.scheduler.sample_timesteps(x0.shape[0], self.generator, self.device)
        xt, sigmas = self.scheduler.add_noise(noise, x0, ts, source, mask)
        gt = self.scheduler.get_velocity(noise, x0, source, mask)

        return {
            "height": target.shape[-2],
            "width": target.shape[-1],
            "timesteps": ts,
            "sigmas": sigmas,
            "noise": noise,
            "gt": gt,
            "mask": mask,
            "edge": edge,
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
        mask = processed_data["mask"]
        edge = processed_data["edge"]
        mask4d = copy.deepcopy(mask)
        edge4d = copy.deepcopy(edge)

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
        mask = self.encode_mask(mask)
        edge = self.encode_mask(edge)
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
        mask = QwenImageEditPlusPipeline._pack_latents(
            mask, mask.shape[0], mask.shape[1], mask.shape[-2], mask.shape[-1]
        )
        edge = QwenImageEditPlusPipeline._pack_latents(
            edge, edge.shape[0], edge.shape[1], edge.shape[-2], edge.shape[-1]
        )

        return {
            "height": target.shape[-2],
            "width": target.shape[-1],
            "noise": noise,
            "mask": mask,
            "edge": edge,
            "mask4d": mask4d,
            "edge4d": edge4d,
            "target": processed_data["raw"]["target"].to(self.device, dtype=self.dtype),
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
        mask: torch.Tensor | None = None,
        edge: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Compute loss without masks and weights"""
        loss_field = F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
        loss_dict = {"loss": 0}

        if mask is not None:
            mask_field = mask * loss_field

            if self.mask_loss_weight > 0:
                total_area = mask.shape[1]
                mask_area = mask.sum()
                mask_field = self.mask_loss_weight * (mask_area / total_area) * mask_field

            mask_loss = (mask_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
            loss_dict["mask_loss"] = mask_loss
            loss_dict["loss"] = loss_dict["loss"] + mask_loss

        if edge is not None and self.edge_loss_weight > 0:
            edge_field = self.edge_loss_weight * edge * loss_field
            edge_loss = (edge_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
            loss_dict["edge_loss"] = edge_loss
            loss_dict["loss"] = loss_dict["loss"] + edge_loss

        if loss_dict["loss"] == 0:
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
        loss = self.compute_loss(predictions, inputs["gt"], mask=inputs["mask"], edge=inputs["edge"])
        return loss

    @torch.inference_mode()
    def eval_step(self, batch, global_step, num_inference_steps: int = 50, cfg_scale: float = 0) -> list:
        from tqdm import tqdm

        inputs = self.prepare_eval_inputs(batch, cfg_scale)
        xt = inputs["noise"]
        source = inputs["conditions"][0]
        mask = inputs["mask"]
        mask4d = inputs["mask4d"]
        edge4d = inputs["edge4d"]

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
        return [mask4d, edge4d, output]
