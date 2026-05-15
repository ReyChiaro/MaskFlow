import os
import math
import torch
import torchvision.transforms.functional as T

from typing import Any

from data_module.dataset import SchemaDataset
from data_module.utils import ASPECT_RATIOS


class MaskEditDataset(SchemaDataset):

    def __init__(
        self,
        image_root,
        data_file,
        data_load_ratio=1,
        max_resolution: int = 1024 * 1024,
        divisible_by: int = 16,
        enable_aspect_bucket=False,
        enable_prompt_truncation: bool = False,
        replace_prompt_placeholder_with: str | None = None,
    ):
        super().__init__(image_root, data_file, data_load_ratio, enable_aspect_bucket)

        self.divisible_by = divisible_by
        self.max_resolution = max_resolution
        self.replace_prompt_placeholder_with = replace_prompt_placeholder_with
        self.enable_prompt_truncation = enable_prompt_truncation

    def _truncate_prompt(self, prompt: str) -> str:
        for separator in (".", "!", "?", "。", "！", "？"):
            if separator in prompt:
                return prompt.split(separator, 1)[0].strip()
        return prompt.strip()

    def _replace_placeholder(self, prompt: str, replacement: str, placeholder: str = "[TARGET]"):
        return prompt.replace(placeholder, replacement)

    def _nearest_aspect_ratio(self, image: torch.Tensor, aspect_ratios: list[str] = ASPECT_RATIOS) -> str:
        return min(
            aspect_ratios,
            key=lambda x: abs(image.shape[-1] / image.shape[-2] - int(x.split(":")[0]) / int(x.split(":")[1])),
        )

    def _crop_image_to_aspect_ratio(
        self,
        image: torch.Tensor,
        aspect_ratios: list[str] = ASPECT_RATIOS,
    ) -> tuple[torch.Tensor, tuple[int]]:
        # ---------------- Reshape to aspect ---------------- #
        aspect = self._nearest_aspect_ratio(image, aspect_ratios)
        aspect_ratio = int(aspect.split(":")[0]) / int(aspect.split(":")[1])
        org_h, org_w = image.shape[-2:]
        org_aspect = org_w / org_h
        h, w = (org_h, int(org_h * aspect_ratio)) if org_aspect > aspect_ratio else (int(org_w / aspect_ratio), org_w)
        image = T.center_crop(image, [h, w])
        return image, aspect_ratio

    def _reshape_to_divisible_max_resolution(self, image: torch.Tensor, aspect_ratio: float | None = None):
        aspect_ratio = aspect_ratio or image.shape[-1] / image.shape[-2]
        # ------------- Reshape to max resolution ------------- #
        max_h = int(math.sqrt(self.max_resolution / aspect_ratio))
        max_w = int(math.sqrt(self.max_resolution * aspect_ratio))
        max_h = max_h // self.divisible_by * self.divisible_by
        max_w = max_w // self.divisible_by * self.divisible_by
        image = T.resize(image, [max_h, max_w])
        return image

    def _preprocess_prompt(self, prompt, **kwargs) -> str:
        if self.enable_prompt_truncation:
            prompt = self._truncate_prompt(prompt)
        if self.replace_prompt_placeholder_with is not None:
            prompt = self._replace_placeholder(prompt, self.replace_prompt_placeholder_with)
        return prompt

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % self.num_samples]
        prompt = sample["prompt"]
        conditions = sample["conditions"]
        target = sample["target"]

        conditions: list[torch.Tensor] = [self._load_image_tensor(os.path.join(self.image_root, c)) for c in conditions]
        target: torch.Tensor = self._load_image_tensor(os.path.join(self.image_root, target))

        # Reshape conditions and target
        target, aspect_ratio = self._crop_image_to_aspect_ratio(target, ASPECT_RATIOS)
        target = self._reshape_to_divisible_max_resolution(target, aspect_ratio)
        conditions = [self._reshape_to_divisible_max_resolution(c, aspect_ratio) for c in conditions]

        return {
            "prompt": self._preprocess_prompt(prompt),
            "conditions": self._preprocess_conditions(conditions),
            "target": self._preprocess_target(target),
        }
