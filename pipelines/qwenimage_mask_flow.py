import copy
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_dimensions,
    CONDITION_IMAGE_SIZE,
    QwenImageEditPlusPipeline,
)
from typing import Literal

from pipelines.qwenimage_edit_plus import QwenImageEditPlusManager


class QwenImageMaskFlowManager(QwenImageEditPlusManager):

    def __init__(
        self,
        pretrained_model_name_or_path,
        dilation_kernel_size: int,
        blur_kernel_size: int,
        blur_sigmas: float,
        xt_unmask_strategy: Literal["x0", "x1", "noisy_x0", "noisy_x1"],
        scheduler_configs=None,
    ):
        super().__init__(pretrained_model_name_or_path, scheduler_configs)
        self.dilation_kernel_size = dilation_kernel_size
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigmas = blur_sigmas

        self.xt_unmask_strategy = xt_unmask_strategy

    def dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        padding = self.dilation_kernel_size // 2
        dilated = F.max_pool2d(mask, kernel_size=self.dilation_kernel_size, padding=padding, stride=1)
        return dilated

    def erode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = 1.0 - mask
        padding = self.dilation_kernel_size // 2
        eroded = F.max_pool2d(mask, kernel_size=self.dilation_kernel_size, padding=padding, stride=1)
        return 1.0 - eroded

    def blur_mask(self, mask: torch.Tensor) -> torch.Tensor:
        blur_mask_tensor = T.gaussian_blur(mask, kernel_size=self.blur_kernel, sigma=self.blur_sigma)
        blur_mask_tensor[~mask] = blur_mask_tensor[~mask] * 2
        blur_mask_tensor[mask] = 1
        return blur_mask_tensor

    def add_noise(
        self,
        eps: torch.Tensor,
        x0: torch.Tensor,
        conditions: list[torch.Tensor],
        sigmas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = conditions[0]
        m = conditions[1]
        mask_xt = m * self.scheduler.add_noise(sigmas, eps, x0)
        unmask_xt = (1 - m) * x0
        target_vt = (m * eps + (1 - m) * x0) - (m * x0 + (1 - m) * x0)

        if self.xt_unmask_strategy == "x0":
            unmask_xt = (1 - m) * x0
            target_vt = (m * eps + (1 - m) * x0) - (m * x0 + (1 - m) * x0)
        elif self.xt_unmask_strategy == "x1":
            unmask_xt = (1 - m) * x1
            target_vt = (m * eps + (1 - m) * x1) - (m * x0 + (1 - m) * x1)
        elif self.xt_unmask_strategy == "noisy_x0":
            unmask_xt = (1 - m) * ((1 - sigmas) * x0 + sigmas * eps)
            target_vt = (m * eps + (1 - m) * eps) - (m * x0 + (1 - m) * x0)
        elif self.xt_unmask_strategy == "noisy_x1":
            unmask_xt = (1 - m) * ((1 - sigmas) * x1 + sigmas * eps)
            target_vt = (m * eps + (1 - m) * eps) - (m * x0 + (1 - m) * x1)
        xt = mask_xt + unmask_xt
        return xt, target_vt

    def preprocess_everything(self, batch, device, dtype):
        r"""
        Preprocess batched data, conduct reshape or any other augmentations on it.
        """
        prompt: str | list[str] = batch["prompt"]
        source: torch.Tensor = batch["conditions"][0].to(device, dtype=dtype)
        hard_mask: torch.Tensor = batch["conditions"][1].to(device, dtype=dtype)
        target: torch.Tensor = batch["target"].to(device, dtype=dtype)

        # ---------------- Preprocess ---------------- #
        # To tensor and reshape to target areas
        _, _, H, W = target.shape
        tgt_aspect = W / H
        cond_H, cond_W = calculate_dimensions(CONDITION_IMAGE_SIZE, tgt_aspect)

        target = self.text_pipeline.image_processor.preprocess(target, H, W).unsqueeze(2)
        source = self.text_pipeline.image_processor.preprocess(source, H, W).unsqueeze(2)
        hard_mask = self.text_pipeline.image_processor.resize(hard_mask, H, W)

        hard_edge = self.dilate_mask(hard_mask) - self.erode_mask(hard_mask)
        soft_edge = self.blur_mask(hard_edge)
        hard_mask = self.dilate_mask(hard_mask)
        soft_mask = self.blur_mask(hard_mask)

        small_conditions = [
            self.text_pipeline.image_processor.resize(source, cond_H, cond_W),
            self.text_pipeline.image_processor.resize(soft_mask, cond_H, cond_W),
        ]

        batch["prompt"] = prompt
        batch["conditions"] = [source, soft_mask]
        batch["target"] = target.to(device, dtype=dtype)
        batch["edge_mask"] = soft_edge
        batch["conditions_to_text_pipeline"] = [c.to(device, dtype=dtype) for c in small_conditions]

        return batch

    def compute_loss(self, predictions: torch.Tensor, ground_truths: torch.Tensor) -> torch.Tensor:
        r"""Compute loss without masks and weights"""
        return (
            F.mse_loss(predictions.float(), ground_truths.float(), reduction="none")
            .reshape(predictions.shape[0], -1)
            .mean(dim=1)
        ).mean()

    def forward_step(self, batch, generator=None, device=None, dtype=None):
        batch = self.preprocess_everything(batch, device, dtype)
        inputs = self.encode_everything(batch, generator, device, dtype)
        noisy_latents, truth_latents = self.add_noise(
            inputs["noise"], inputs["target_latents"], inputs["conditions"], inputs["sigmas"]
        )
        hidden_states = torch.cat([noisy_latents] + [c for c in inputs["conditions"]], dim=1)
        predictions = self.denoise(
            hidden_states,
            inputs["timesteps"],
            inputs["prompt_embeds"],
            inputs["prompt_embeds_mask"],
            inputs["image_shapes"],
            noisy_latents.shape[1],
        )
        loss = self.compute_loss(predictions, truth_latents)
        return loss
