import os
import json
import torch
import argparse
import torchvision.transforms.functional as T

from pathlib import Path
from PIL import Image
from loguru import logger

from evaluator.evaluator import Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Metric Calculation")
    parser.add_argument("--source", type=str, help="Path to Dir or Image.")
    parser.add_argument("--target", type=str, help="Path to Dir or Image.")
    parser.add_argument("--mask", type=str, default=None, help="Path to Dir or Image for foreground masks.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--save-to", type=str, help="JSON file for outputs.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.rank}") if args.rank >= 0 else torch.device("cpu")
    evaluator = Evaluator(device)

    metric_content = "\n" + " Metrics ".center(70, "=")
    for k, fn in evaluator.metrics.items():
        metric_name_len = len(k)
        metric_content += f"\n  {k}" + " " * (25 - metric_name_len)
        metric_content += str(fn)
    metric_content += "\n" + "=" * 70
    logger.info(metric_content)

    def collect_images(path: str) -> list[str]:
        return [path] if Path(path).is_file() else [os.path.join(path, i) for i in sorted(os.listdir(path))]

    source = collect_images(args.source)
    target = collect_images(args.target)
    mask = collect_images(args.mask) if args.mask is not None else None
    save_to = args.save_to

    assert len(source) == len(target), f"Num of source and target should be equal."
    if mask is not None:
        assert len(source) == len(mask), f"Num of source and mask should be equal."

    config_content = "\n" + " Configs ".center(70, "=")
    config_content += f"\n  device: {device}"
    config_content += f"\n  source: {source}"
    config_content += f"\n  target: {target}"
    config_content += f"\n  mask: {mask}"
    config_content += f"\n  save_to: {save_to}"
    config_content += "\n" + "=" * 70
    logger.info(config_content)

    source = torch.stack([T.to_tensor(Image.open(s).convert("RGB")) for s in source], dim=0)
    target = torch.stack([T.to_tensor(Image.open(t).convert("RGB")) for t in target], dim=0)
    mask = torch.stack([T.to_tensor(Image.open(m).convert("L")) for m in mask], dim=0) if mask is not None else None

    metric_results = evaluator.compute(source, target, mask=mask)
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
