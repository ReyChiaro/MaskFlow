import os
import math
import json
import torch
import torchvision.transforms.functional as T

from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from typing import Any


class SchemaDataset(Dataset):

    def __init__(
        self,
        image_root: str,
        data_file: str,
        load_start: int | float = 0.0,
        load_end: int | float = 1.0,
    ):
        r"""
        Args:
            data_file (str): JSONL file.

        TODO: Add aspect ratio bucket
        """
        super().__init__()
        self.image_root = image_root
        self.data_file = data_file

        self.samples = self._load_data_file(load_start=load_start, load_end=load_end)
        self.num_samples = len(self.samples)

    def _load_data_file(self, load_start: int | float = 0.0, load_end: int | float = 1.0):
        if not os.path.exists(self.data_file):
            raise FileExistsError(f"{self.data_file} is not exists.")

        samples = []
        with open(self.data_file, "r") as f:
            for l in f:
                samples.append(json.loads(l))

        num_total = len(samples)
        start_idx = load_start if isinstance(load_start, int) else math.floor(load_start * num_total)
        end_idx = load_end if isinstance(load_end, int) else math.floor(load_end * num_total) + 1
        return samples[start_idx : min(len(samples), end_idx)]

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        return T.to_tensor(Image.open(path).convert("RGB"))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % self.num_samples]
        prompt = sample["prompt"]
        negative_prompt = sample.get("negative_prompt", "")
        conditions = sample["conditions"]
        target = sample["target"]
        image_name = Path(target).stem

        conditions = [self._load_image_tensor(os.path.join(self.image_root, c)) for c in conditions]
        target = self._load_image_tensor(os.path.join(self.image_root, target))

        return {
            "image_name": image_name,
            "prompt": self._preprocess_prompt(prompt),
            "negative_prompt": negative_prompt,
            "conditions": self._preprocess_conditions(conditions),
            "target": self._preprocess_target(target),
        }

    def _preprocess_prompt[T](self, prompt: T, **kwargs) -> T:
        return prompt

    def _preprocess_conditions[T](self, conditions: T, **kwargs) -> T:
        return conditions

    def _preprocess_target[T](self, target: T, **kwargs) -> T:
        return target
