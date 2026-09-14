import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torchvision.transforms.v2.functional as T

from data_module.dataset import SchemaDataset
from data_module.sample_utils import image_name
from data_module.utils import (
    center_crop_to_aspect_ratio,
    crop_image_to_aspect_ratio,
    reshape_to_divisible_max_resolution,
)


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
        target = sample["target"]
        target = self._load_image_tensor(os.path.join(self.image_root, target)) if target is not None else None
        conditions = sample["conditions"]
        if isinstance(conditions, dict):
            conditions = {k: self._load_image_tensor(os.path.join(self.image_root, c)) for k, c in conditions.items()}
        else:
            conditions = [self._load_image_tensor(os.path.join(self.image_root, c)) for c in conditions]
        return self._preprocess_sample(sample, target, conditions, image_name(sample))

    def _preprocess_sample(self, sample, target, conditions, name) -> dict[str, Any]:
        prompt = sample["prompt"]
        negative_prompt = sample.get("negative_prompt", "")
        edit_instruction = sample.get("edit_instruction", "")

        # Reshape conditions and target
        reference = target if target is not None else conditions["source"]
        reference, aspect_ratio = crop_image_to_aspect_ratio(reference)
        if target is not None:
            target = reshape_to_divisible_max_resolution(reference, aspect_ratio, divisible_by=self.divisible_by)

        if isinstance(conditions, dict):
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
            conditions = [center_crop_to_aspect_ratio(c, aspect_ratio) for c in conditions]
            conditions = [
                reshape_to_divisible_max_resolution(c, aspect_ratio, divisible_by=self.divisible_by) for c in conditions
            ]

        return {
            "image_name": name,
            "prompt": self._preprocess_prompt(prompt),  # Prompt without position cues
            "negative_prompt": negative_prompt,
            "edit_instruction": edit_instruction,  # Prompt with position cues
            "conditions": self._preprocess_conditions(conditions),
            "target": self._preprocess_target(target) if target is not None else None,
        }


class HFMaskEditDataset(MaskEditDataset):
    """Load downloaded MaskEdit-10k Parquet shards from data_root/data/<subset>.

    Subsets are concatenated in the supplied order, with shards sorted by name.
    load_start/load_end slice the combined samples using MaskEditDataset semantics.
    Only metadata and row locations are indexed eagerly; each worker caches one
    row group of embedded images at a time.
    """

    SUBSETS = ("scene", "infographics_en", "infographics_cn")
    COLUMNS = ("source", "mask", "target", "prompt", "negative_prompt", "edit_instruction")

    def __init__(
        self,
        data_root: str,
        subsets: str | list[str] = "scene",
        split: str = "train",
        load_start: int | float = 0.0,
        load_end: int | float = 1.0,
        divisible_by: int = 32,
        enable_prompt_truncation: bool = False,
        replace_prompt_placeholder_with: str | None = None,
    ):
        self.subsets = [subsets] if isinstance(subsets, str) else list(subsets)
        if not self.subsets or any(subset not in self.SUBSETS for subset in self.subsets):
            raise ValueError(f"subsets must contain one or more of {self.SUBSETS}, got {self.subsets}.")
        if len(set(self.subsets)) != len(self.subsets):
            raise ValueError(f"Duplicate subsets: {self.subsets}.")
        if split not in ("train", "test"):
            raise ValueError(f"split must be train or test, got {split!r}.")

        self.data_root = Path(data_root)
        self.split = split
        self._cached_key = None
        self._cached_table = None
        super().__init__(
            image_root=data_root,
            data_file=None,
            load_start=load_start,
            load_end=load_end,
            divisible_by=divisible_by,
            enable_prompt_truncation=enable_prompt_truncation,
            replace_prompt_placeholder_with=replace_prompt_placeholder_with,
        )
        # if not self.num_samples:
        #     raise ValueError("No samples selected; check the Parquet files and load_start/load_end.")

    def _load_data_file(self, load_start: int | float = 0.0, load_end: int | float = 1.0):
        samples = []
        for subset in self.subsets:
            directory = self.data_root / "data" / subset
            files = sorted(directory.glob(f"{self.split}-*.parquet"))
            if not files:
                raise FileNotFoundError(f"No {self.split} Parquet shards found in {directory}.")
            for path in files:
                with pq.ParquetFile(path) as parquet:
                    missing = set(self.COLUMNS) - set(parquet.schema_arrow.names)
                    if missing:
                        raise ValueError(f"Missing columns in {path}: {sorted(missing)}.")
                    # Keep text and filenames available without loading image bytes.
                    columns = [f"{role}.path" for role in ("source", "mask", "target")]
                    columns += [name for name in parquet.schema_arrow.names if name not in ("source", "mask", "target")]
                    records = parquet.read(columns=columns).to_pylist()
                    offset = 0
                    for group in range(parquet.num_row_groups):
                        count = parquet.metadata.row_group(group).num_rows
                        for row in range(count):
                            sample = records[offset + row]
                            sample["conditions"] = {role: sample.pop(role)["path"] for role in ("source", "mask")}
                            sample["target"] = sample["target"]["path"]
                            sample.update(
                                {
                                    "parquet_file": path,
                                    "row_group": group,
                                    "row_index": row,
                                }
                            )
                            samples.append(sample)
                        offset += count

        num_total = len(samples)
        start_idx = load_start if isinstance(load_start, int) else math.floor(load_start * num_total)
        end_idx = load_end if isinstance(load_end, int) else math.floor(load_end * num_total) + 1
        return samples[start_idx : min(len(samples), end_idx)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_key"] = None
        state["_cached_table"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.samples[index % self.num_samples]
        key = (entry["parquet_file"], entry["row_group"])
        try:
            table = self._read_row_group(entry)
            sample = table.slice(entry["row_index"], 1).to_pylist()[0]
            target = self._load_image_tensor(BytesIO(sample["target"]["bytes"]))
            conditions = {k: self._load_image_tensor(BytesIO(sample[k]["bytes"])) for k in ("source", "mask")}
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"Failed to load {key[0]}, row group {key[1]}, row {entry['row_index']}: {error}"
            ) from error
        return self._preprocess_sample(sample, target, conditions, image_name(entry))

    def _read_row_group(self, entry):
        key = (entry["parquet_file"], entry["row_group"])
        if key != self._cached_key:
            with pq.ParquetFile(key[0]) as parquet:
                table = parquet.read_row_group(key[1], columns=list(self.COLUMNS))
            self._cached_table = table
            self._cached_key = key
        return self._cached_table

    def load_image_bytes(self, index: int, role: str) -> bytes:
        """Read an original image for consumers that apply their own preprocessing."""
        if role not in ("source", "mask", "target"):
            raise ValueError(f"Unknown image role: {role!r}.")
        entry = self.samples[index % self.num_samples]
        try:
            table = self._read_row_group(entry)
            return table.column(role)[entry["row_index"]].as_py()["bytes"]
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"Failed to read {role} from {entry['parquet_file']}, "
                f"row group {entry['row_group']}, row {entry['row_index']}: {error}"
            ) from error
