import os
import json
import torch
import dataclasses
import torchvision.transforms.functional as T

from torch.utils.data import Dataset
from PIL import Image
from loguru import logger
from typing import Any

from data_module.utils import is_bucketed_dataset


class SchemaDataset(Dataset):

    def __init__(
        self,
        data_file: str,
        data_load_ratio: float = 1.0,
        enable_aspect_bucket: bool = False,
    ):
        r"""
        Args:
            data_file (str): JSON or JSONL file.
        """
        super().__init__()
        self.data_file = data_file
        self.is_bucketed = is_bucketed_dataset(self.data_file)

        # TODO: Add aspect ratio bucket
        if self.is_bucketed:
            logger.error(f"Current dataset not support aspect ratio buckets.")

        self.samples = self._load_data_file(load_ratio=data_load_ratio)
        self.num_samples = len(self.samples)

    def _load_data_file(self, load_ratio: float = 1.0):
        if not os.path.exists(self.data_file):
            raise FileExistsError(f"{self.data_file} is not exists.")

        with open(self.data_file, "r") as f:
            samples: list[dict[str, Any]] = json.load(f)

        return samples[:load_ratio]

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        return T.to_tensor(Image.open(path).convert("RGB"))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index % self.num_samples]
        prompt = sample["prompt"]
        conditions = sample["conditions"]
        target = sample["target"]

        conditions = [self._load_image_tensor(c) for c in conditions]
        target = self._load_image_tensor(target)

        return {
            "prompt": self._preprocess_prompt(prompt),
            "conditions": self._preprocess_conditions(conditions),
            "target": self._preprocess_target(target),
        }

    def _preprocess_prompt[T](self, prompt: T, **kwargs) -> T:
        return prompt

    def _preprocess_conditions[T](self, conditions: T, **kwargs) -> T:
        return conditions

    def _preprocess_target[T](self, target: T, **kwargs) -> T:
        return target
