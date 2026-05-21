import copy
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
        # mask = self.image_processor.preprocess(mask, h, w).unsqueeze(2)
        # edge = self.image_processor.preprocess(edge, h, w).unsqueeze(2)
        mask = self.image_processor.resize(mask, h, w)
        edge = self.image_processor.resize(edge, h, w)

        return {
            "raw": raw_batch,
            "prompt": prompt,
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
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            processed_data["prompt"], processed_data["conditions_vlm"]
        )

        image_shapes = []
        conditions_dit = processed_data["conditions_dit"]
        target = processed_data["target"]
        mask = processed_data["mask"]
        edge = processed_data["edge"]

        # ---------------- Encode and Pack ---------------- #
        # Encode
        tgt = self.encode_image(target, sample_mode)
        # mask = self.encode_image(mask, sample_mode)
        # edge = self.encode_image(edge, sample_mode)
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

    def prepare_eval_inputs(self, batch):
        r"""
        Prepare training evaluation inputs.
        The sample mode for VAE is fixed to `argmax`, `target` must be provided.
        """
        sample_mode = "argmax"
        processed_data = self.preprocess_inputs(batch)
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            processed_data["prompt"], processed_data["conditions_vlm"]
        )

        image_shapes = []
        conditions_dit = processed_data["conditions_dit"]
        target = processed_data["target"]
        mask = processed_data["mask"]
        edge = processed_data["edge"]

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
        # mask = self.encode_image(mask, sample_mode)
        # edge = self.encode_image(edge, sample_mode)
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
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
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
        if mask is not None and self.mask_loss_weight > 0:
            loss_field = mask * loss_field
            loss = (loss_field.reshape(predictions.shape[0], -1).mean(dim=1)).mean()
            # total_area = mask.shape[1]
            # mask_area = mask.sum()
            # mask_loss = mask * loss_field
            # mask_loss = self.mask_loss_weight

            # loss = mask_loss
            pass

        if edge is not None and self.edge_loss_weight > 0:
            pass

        return loss

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
    def eval_step(self, batch, num_inference_steps: int = 50, cfg: float = 0.0):
        r"""
        TODO: Support CFG
        """
        from tqdm import tqdm
        from torchvision.utils import save_image
        from PIL import Image

        inputs = self.prepare_eval_inputs(batch)

        xt = inputs["noise"]
        source = inputs["conditions"][0]
        mask = inputs["mask"]
        edge = inputs["edge"]

        source_image = QwenImageEditPlusPipeline._unpack_latents(
            source, inputs["height"], inputs["width"], self.vae_scale_factor
        )
        source_image = self.decode_image(source_image)
        mask_image = QwenImageEditPlusPipeline._unpack_latents(
            mask, inputs["height"], inputs["width"], self.vae_scale_factor
        )[:, :, 0]
        edge_image = QwenImageEditPlusPipeline._unpack_latents(
            edge, inputs["height"], inputs["width"], self.vae_scale_factor
        )[:, :, 0]
        mask_image = mask_image.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        edge_image = edge_image.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        mask_image = F.interpolate(mask_image, [inputs["height"], inputs["width"]])
        edge_image = F.interpolate(edge_image, [inputs["height"], inputs["width"]])

        colored_mask = copy.deepcopy(mask_image)
        colored_mask[:, 1, ...] = 0
        colored_mask[:, 2, ...] = 0
        masked_source = torch.where(mask_image >= 1, 0.4 * source_image + 0.6 * colored_mask, source_image)

        colored_edge = copy.deepcopy(edge_image)
        colored_edge[:, 1, ...] = 0
        colored_edge[:, 2, ...] = 0
        edgeed_source = torch.where(edge_image >= 1, 0.4 * source_image + 0.6 * colored_edge, source_image)

        step = 0
        with self.scheduler.inference_sampler(xt, num_inference_steps, source, mask, xt.shape[1]) as sampler:
            for xt, t, inferencer in tqdm(sampler, total=num_inference_steps):
                step += 1
                hidden_states = torch.cat([xt] + [c for c in inputs["conditions"]], dim=1)
                pred = self.denoise(
                    hidden_states,
                    t.expand(hidden_states.shape[0]).to(device=self.device, dtype=self.dtype),
                    inputs["prompt_embeds"],
                    inputs["prompt_embeds_mask"],
                    inputs["image_shapes"],
                    xt.shape[1],
                )
                inferencer.step(pred)

                output = QwenImageEditPlusPipeline._unpack_latents(
                    xt, inputs["height"], inputs["width"], self.vae_scale_factor
                )
                output = self.decode_image(output)

                tensor_to_save = torch.cat([source_image, mask_image, output], dim=-1)
                masked_predict = torch.where(mask_image > 0, 0.4 * output + 0.6 * colored_mask, output)
                mask_tensor_to_save = torch.cat([masked_source, mask_image, masked_predict], dim=-1)

                edgeed_predict = torch.where(edge_image > 0, 0.4 * output + 0.6 * colored_edge, output)
                edge_tensor_to_save = torch.cat([edgeed_source, edge_image, edgeed_predict], dim=-1)

                save_image(
                    torch.cat([tensor_to_save, mask_tensor_to_save, edge_tensor_to_save], dim=-2),
                    f"outputs/experiments/project-train/_test_infer/infer-step-{step}.jpg",
                )

        # Gen gif
        pil_images = [
            Image.open(f"outputs/experiments/project-train/_test_infer/infer-step-{s}.jpg")
            .convert("RGB")
            .resize([inputs["height"], inputs["width"]])
            for s in range(1, num_inference_steps + 1)
        ]
        pil_images[0].save(
            "outputs/experiments/project-train/_test_infer/denoise.gif",
            save_all=True,
            append_images=pil_images[1:],
            duration=100,
            loop=0,
        )

        output = QwenImageEditPlusPipeline._unpack_latents(xt, inputs["height"], inputs["width"], self.vae_scale_factor)
        output = self.decode_image(output)
        return output
