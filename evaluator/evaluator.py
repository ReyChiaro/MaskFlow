import torch

from dataclasses import dataclass
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

    def __init__(self, device: torch.device = torch.device("cpu")):
        initialize_metrics()
        self.device = device
        self.metrics = get_metrics()

    def compute(
        self,
        sources: torch.Tensor | list[torch.Tensor],
        targets: torch.Tensor | list[torch.Tensor],
        **kwargs,
    ) -> dict[str, float]:
        assert _num_images(sources) == _num_images(
            targets
        ), f"Given sources ({_num_images(sources)}) and targets ({_num_images(targets)}) should contain same num of items."

        sources = _to_device(sources, self.device)
        targets = _to_device(targets, self.device)
        kwargs = {k: _to_device(v, self.device) for k, v in kwargs.items()}
        has_mask = kwargs.get("mask") is not None

        results = {}
        for metric_name, metric_fn in self.metrics.items():
            if metric_name.endswith(("-FG", "-BG")) and not has_mask:
                logger.info(f"Skip {metric_name}: mask is not provided.")
                continue
            logger.info(f"Evaluate {metric_name}")
            result: float = metric_fn(sources, targets, **kwargs)
            results[metric_name] = result
        return results
