import json
import glob
import torch
import argparse
import torchvision.transforms.functional as T

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from loguru import logger

from evaluator.evaluator import Evaluator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(path: str) -> list[str]:
    paths = sorted(glob.glob(path)) if glob.has_magic(path) else [path]
    if not paths:
        raise FileNotFoundError(f"No paths matched: {path}")

    images = []
    for item in paths:
        item_path = Path(item)
        if item_path.is_file():
            images.append(str(item_path))
            continue
        if not item_path.is_dir():
            raise FileNotFoundError(f"Path does not exist or is not a file/directory: {item}")

        images.extend(
            str(child)
            for child in sorted(item_path.iterdir())
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not images:
        raise FileNotFoundError(f"No image files found for: {path}")
    images.sort(key=lambda image_path: (Path(image_path).stem, image_path))
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Metric Calculation")
    parser.add_argument("--source", type=str, help="Path to Dir or Image.")
    parser.add_argument("--target", type=str, help="Path to Dir or Image.")
    parser.add_argument("--mask", type=str, default=None, help="Path to Dir or Image for foreground masks.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--save-to", type=str, help="JSON file for outputs.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.rank}") if args.rank >= 0 else torch.device("cpu")
    evaluator = Evaluator(device=device)

    metric_content = "\n" + " Metrics ".center(70, "=")
    for k, fn in evaluator.metrics.items():
        metric_name_len = len(k)
        metric_content += f"\n  {k}" + " " * (25 - metric_name_len)
        metric_content += str(fn)
    metric_content += "\n" + "=" * 70
    logger.info(metric_content)

    source = collect_images(args.source)
    target = collect_images(args.target)
    mask = collect_images(args.mask) if args.mask is not None else None
    save_to = args.save_to

    assert len(source) == len(target), f"Num of source and target should be equal."
    if mask is not None:
        assert len(source) == len(mask), f"Num of source and mask should be equal."

    config_content = "\n" + " Configs ".center(70, "=")
    config_content += f"\n  device: {device}"
    config_content += f"\n  source: {len(source)}"
    config_content += f"\n  target: {len(target)}"
    config_content += f"\n  mask: {len(mask)}"
    config_content += f"\n  save_to: {save_to}"
    config_content += "\n" + "=" * 70
    logger.info(config_content)

    logger.info(f"Opening images and converting them to Tensors.")
    source_tensors = []
    target_tensors = []
    mask_tensors = [] if mask is not None else None
    for i in tqdm(range(len(source)), total=len(source), desc="Loading Tensors"):
        assert (
            Path(source[i]).stem == Path(target[i]).stem
        ), f"source {Path(source[i]).stem} not match to target {Path(target[i]).stem}."
        source_image = T.to_tensor(Image.open(source[i]).convert("RGB")).clamp(0.0, 1.0)
        target_image = T.resize(T.to_tensor(Image.open(target[i]).convert("RGB")), [*source_image.shape[-2:]]).clamp(
            0.0, 1.0
        )
        target_tensors.append(target_image)
        source_tensors.append(source_image)
        if mask is not None:
            mask_image = T.to_tensor(Image.open(mask[i]).convert("L")).clamp(0.0, 1.0)
            mask_tensors.append(mask_image)

    metric_results = evaluator.compute(source_tensors, target_tensors, mask=mask_tensors)
    logger.info(f"Metric calculation finished.")

    metric_content = "\n" + " Results ".center(70, "=")
    for name, result in metric_results.items():
        metric_name_len = len(name)
        metric_content += f"\n  {name}" + " " * (25 - metric_name_len)
        metric_content += str(result)
    metric_content += "\n" + "=" * 70
    logger.info(metric_content)

    with open(save_to, "w") as f:
        json.dump(metric_results, f, indent=4, sort_keys=True)
    logger.info(f"Metric results saved to {save_to}.")
