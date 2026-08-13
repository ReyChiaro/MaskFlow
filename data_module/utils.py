import os
import json
import math
import torch
import torchvision.transforms.v2.functional as T

from loguru import logger

ASPECT_RATIOS = ["1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
MAX_RESOLUTION = 1024 * 1024
MAX_CONDITION_RESOLUTION = 384 * 384
DIVISIBLE_BY = 32


def nearest_aspect_ratio(image: torch.Tensor, aspect_ratios: list[str] = ASPECT_RATIOS) -> str:
    return min(
        aspect_ratios,
        key=lambda x: abs(image.shape[-1] / image.shape[-2] - int(x.split(":")[0]) / int(x.split(":")[1])),
    )


def crop_image_to_aspect_ratio(
    image: torch.Tensor,
    aspect_ratios: list[str] = ASPECT_RATIOS,
) -> tuple[torch.Tensor, tuple[int]]:
    # ---------------- Reshape to aspect ---------------- #
    aspect = nearest_aspect_ratio(image, aspect_ratios)
    aspect_ratio = int(aspect.split(":")[0]) / int(aspect.split(":")[1])
    return center_crop_to_aspect_ratio(image, aspect_ratio), aspect_ratio


def center_crop_to_aspect_ratio(image: torch.Tensor, aspect_ratio: float) -> torch.Tensor:
    """Center-crop an aligned image to an already selected aspect ratio."""
    org_h, org_w = image.shape[-2:]
    org_aspect = org_w / org_h
    h, w = (org_h, int(org_h * aspect_ratio)) if org_aspect > aspect_ratio else (int(org_w / aspect_ratio), org_w)
    return T.center_crop(image, [h, w])


def reshape_to_divisible_max_resolution(
    image: torch.Tensor,
    aspect_ratio: float | None = None,
    max_resolution: int = MAX_RESOLUTION,
    divisible_by: int = DIVISIBLE_BY,
    interpolation: T.InterpolationMode = T.InterpolationMode.LANCZOS,
):
    aspect_ratio = aspect_ratio or image.shape[-1] / image.shape[-2]
    # ------------- Reshape to max resolution ------------- #
    max_h = round(math.sqrt(max_resolution / aspect_ratio) / divisible_by) * divisible_by
    max_w = round(math.sqrt(max_resolution * aspect_ratio) / divisible_by) * divisible_by
    image = T.resize(
        image,
        [max_h, max_w],
        interpolation=interpolation,
        antialias=interpolation
        in {
            T.InterpolationMode.BILINEAR,
            T.InterpolationMode.BICUBIC,
            T.InterpolationMode.LANCZOS,
        },
    )
    return image


def is_bucketed_dataset(data_file: str, sample_size: int = 1):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found.")
    ext = os.path.splitext(data_file)[-1].lower()
    assert ext in [".json", ".jsonl"], f"Unsupported data format: {ext}, only JSON or JSONL is supported."

    def _is_bucket_format(data):
        if not isinstance(data, dict):
            return False

        if len(data) == 0:
            return False

        first_key = next(iter(data))
        first_value = data[first_key]

        if "prompt" in data:
            return False

        if isinstance(first_value, list) and len(first_value) > 0:
            if isinstance(first_value[0], dict) and "prompt" in first_value[0]:
                return True

        return False

    if ext == ".jsonl":
        with open(data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                line_data = json.loads(line.strip())
                if _is_bucket_format(line_data):
                    return True
        return False

    elif ext == ".json":
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return _is_bucket_format(data)

    else:
        logger.warning(f"Unsupported data format: {ext}, only JSON or JSONL is supported.")
        return False
