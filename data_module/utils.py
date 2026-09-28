import json
import math
import os

import torch
import torchvision.transforms.functional as T
from loguru import logger
from PIL import Image

ASPECT_RATIOS = ("1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9")


def nearest_aspect_ratio(
    image: torch.Tensor,
    aspect_ratios: tuple[str, ...] = ASPECT_RATIOS,
) -> str:
    return min(
        aspect_ratios,
        key=lambda x: abs(image.shape[-1] / image.shape[-2] - int(x.split(":")[0]) / int(x.split(":")[1])),
    )


def crop_image_to_aspect_ratio(
    image: torch.Tensor,
    aspect_ratios: tuple[str, ...] = ASPECT_RATIOS,
) -> tuple[torch.Tensor, float]:
    aspect = nearest_aspect_ratio(image, aspect_ratios)
    aspect_ratio = int(aspect.split(":")[0]) / int(aspect.split(":")[1])
    return center_crop_to_aspect_ratio(image, aspect_ratio), aspect_ratio


def center_crop_to_aspect_ratio(image: torch.Tensor, aspect_ratio: float) -> torch.Tensor:
    """Center-crop an aligned [C, H, W] image to the selected aspect ratio."""
    org_h, org_w = image.shape[-2:]
    org_aspect = org_w / org_h
    h, w = (org_h, int(org_h * aspect_ratio)) if org_aspect > aspect_ratio else (int(org_w / aspect_ratio), org_w)
    return T.center_crop(image, [h, w])


def image_dimensions(aspect_ratio: float, max_resolution: int = 1024 * 1024, divisible_by: int = 32) -> tuple[int, int]:
    """Return (height, width), divisible by the supplied factor and within the pixel area limit."""
    height = math.floor(math.sqrt(max_resolution / aspect_ratio) / divisible_by) * divisible_by
    width = math.floor(math.sqrt(max_resolution * aspect_ratio) / divisible_by) * divisible_by
    return height, width


def resize_image(
    image: torch.Tensor,
    height: int,
    width: int,
    interpolation: Image.Resampling = Image.Resampling.LANCZOS,
) -> torch.Tensor:
    """Resize a CPU float32 [C, H, W] image in [0, 1] through PIL; return the same tensor format."""
    if image.shape[-2:] == (height, width):
        return image
    pil_image = T.to_pil_image(image.mul(255).round().to(torch.uint8))
    return T.to_tensor(pil_image.resize((width, height), resample=interpolation))


def reshape_to_divisible_max_resolution(
    image: torch.Tensor,
    aspect_ratio: float | None = None,
    max_resolution: int = 1024 * 1024,
    divisible_by: int = 32,
    interpolation: Image.Resampling = Image.Resampling.LANCZOS,
) -> torch.Tensor:
    aspect_ratio = aspect_ratio or image.shape[-1] / image.shape[-2]
    height, width = image_dimensions(aspect_ratio, max_resolution, divisible_by)
    return resize_image(image, height, width, interpolation)


def preprocess_images(
    conditions: dict[str, torch.Tensor] | list[torch.Tensor],
    target: torch.Tensor | None = None,
    max_resolution: int = 1024 * 1024,
    divisible_by: int = 32,
    aspect_ratios: tuple[str, ...] = ASPECT_RATIOS,
) -> tuple[dict[str, torch.Tensor] | list[torch.Tensor], torch.Tensor | None]:
    """Align CPU float32 [C, H, W] images in [0, 1]; source determines the shared crop and size."""
    source = conditions["source"] if isinstance(conditions, dict) else conditions[0]
    aspect = nearest_aspect_ratio(source, aspect_ratios)
    aspect_ratio = int(aspect.split(":")[0]) / int(aspect.split(":")[1])
    height, width = image_dimensions(aspect_ratio, max_resolution, divisible_by)
    if target is not None:
        target = resize_image(center_crop_to_aspect_ratio(target, aspect_ratio), height, width)
    if isinstance(conditions, dict):
        conditions = {
            key: resize_image(
                center_crop_to_aspect_ratio(image, aspect_ratio),
                height,
                width,
                Image.Resampling.NEAREST if key == "mask" else Image.Resampling.LANCZOS,
            )
            for key, image in conditions.items()
        }
    else:
        conditions = [
            resize_image(center_crop_to_aspect_ratio(image, aspect_ratio), height, width) for image in conditions
        ]
    return conditions, target


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
