from dataclasses import dataclass

import torch
from loguru import logger

from evaluator.register import get_metrics, initialize_metrics


def _num_images(images: torch.Tensor | list[torch.Tensor]) -> int:
    return len(images) if isinstance(images, list) else len(images)


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
    return value


@dataclass
class Evaluator:
    def __init__(self, metrics: list[str] | None = None, device: torch.device = torch.device("cpu")):
        initialize_metrics()
        self.device = device
        self.metrics = get_metrics()
        self.requested_metrics = metrics

        if metrics is not None:
            self.metrics = {k: fn for k, fn in self.metrics.items() if k in metrics}

    def compute(
        self,
        sources: torch.Tensor | list[torch.Tensor],
        targets: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs,
    ) -> dict[str, float]:
        metrics = self.metrics
        target_free = targets is None
        if target_free:
            targets = kwargs.pop("originals", None)
            if self.requested_metrics is None:
                metrics = {name: metrics[name] for name in ("CLIP-TEXT", "PSNR", "SSIM", "DISTS")}
        needs_reference = any(not name.startswith("CLIP-TEXT") for name in metrics)
        if needs_reference:
            if targets is None:
                raise ValueError("Target-free metrics require originals (source images).")
            assert _num_images(sources) == _num_images(targets), (
                f"Given sources ({_num_images(sources)}) and targets ({_num_images(targets)}) should contain same num of items."
            )

        sources = _to_device(sources, self.device)
        targets = _to_device(targets, self.device)
        kwargs = {k: _to_device(v, self.device) for k, v in kwargs.items()}
        has_mask = kwargs.get("mask") is not None

        results = {}
        for metric_name, metric_fn in metrics.items():
            if metric_name.startswith("CLIP-TEXT") and kwargs.get("prompts") is None:
                if self.requested_metrics is not None or target_free:
                    raise ValueError("CLIP-TEXT requires prompts.")
                continue
            if metric_name.endswith(("-FG", "-BG")) and not has_mask:
                logger.info(f"Skip {metric_name}: mask is not provided.")
                continue
            logger.info(f"Evaluate {metric_name}")
            result: float = metric_fn(sources, targets, **kwargs)
            logger.info(f"🎉 {metric_name}: {result} 🎉")
            results[metric_name] = result
        return results
