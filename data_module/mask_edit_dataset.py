import os
import torch
import torchvision.transforms.functional as T

from pathlib import Path
from typing import Any

from data_module.dataset import SchemaDataset
from data_module.utils import center_crop_to_aspect_ratio, crop_image_to_aspect_ratio, reshape_to_divisible_max_resolution


class MaskEditDataset(SchemaDataset):

    def __init__(
        self,
        image_root,
        data_file,
        load_start: int | float = 0.0,
        load_end: int | float = 1.0,
        divisible_by: int = 32,
        enable_prompt_truncation: bool = False,
        replace_prompt_placeholder_with: str | None = None,
    ):
        super().__init__(image_root, data_file, load_start, load_end)

        self.divisible_by = divisible_by
        self.replace_prompt_placeholder_with = replace_prompt_placeholder_with
        self.enable_prompt_truncation = enable_prompt_truncation

    def _truncate_prompt(self, prompt: str) -> str:
        for separator in (".", "!", "?", "。", "！", "？"):
            if separator in prompt:
                return prompt.split(separator, 1)[0].strip()
        return prompt.strip()

    def _replace_placeholder(self, prompt: str, replacement: str, placeholder: str = "[MASK_AREA]"):
        return prompt.replace(placeholder, replacement)

    def _preprocess_prompt(self, prompt, **kwargs) -> str:
        if self.enable_prompt_truncation:
            prompt = self._truncate_prompt(prompt)
        if self.replace_prompt_placeholder_with is not None:
            prompt = self._replace_placeholder(prompt, self.replace_prompt_placeholder_with)
        return prompt

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % self.num_samples]
        prompt = sample["prompt"]
        negative_prompt = sample.get("negative_prompt", "")
        edit_instruction = sample.get("edit_instruction", "")
        conditions = sample["conditions"]
        target = sample["target"]
        image_name = Path(target).stem

        target: torch.Tensor = self._load_image_tensor(os.path.join(self.image_root, target))

        # Reshape conditions and target
        target, aspect_ratio = crop_image_to_aspect_ratio(target)
        target = reshape_to_divisible_max_resolution(target, aspect_ratio, divisible_by=self.divisible_by)

        if isinstance(conditions, dict):
            conditions = {k: self._load_image_tensor(os.path.join(self.image_root, c)) for k, c in conditions.items()}
            conditions = {k: center_crop_to_aspect_ratio(c, aspect_ratio) for k, c in conditions.items()}
            conditions = {
                k: reshape_to_divisible_max_resolution(
                    c,
                    aspect_ratio,
                    divisible_by=self.divisible_by,
                    interpolation=T.InterpolationMode.NEAREST if k == "mask" else T.InterpolationMode.BICUBIC,
                )
                for k, c in conditions.items()
            }
        else:
            conditions = [self._load_image_tensor(os.path.join(self.image_root, c)) for c in conditions]
            conditions = [center_crop_to_aspect_ratio(c, aspect_ratio) for c in conditions]
            conditions = [
                reshape_to_divisible_max_resolution(c, aspect_ratio, divisible_by=self.divisible_by)
                for c in conditions
            ]

        return {
            "image_name": image_name,
            "prompt": self._preprocess_prompt(prompt),  # Prompt without position cues
            "negative_prompt": negative_prompt,
            "edit_instruction": edit_instruction,       # Prompt with position cues
            "conditions": self._preprocess_conditions(conditions),
            "target": self._preprocess_target(target),
        }
