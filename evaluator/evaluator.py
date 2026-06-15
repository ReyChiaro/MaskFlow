import torch

from dataclasses import dataclass
from loguru import logger

from evaluator.register import get_metrics, initialize_metrics


@dataclass
class Evaluator:

    def __init__(self, device: torch.device = torch.device("cpu")):
        initialize_metrics()
        self.device = device
        self.metrics = get_metrics()

    def compute(
        self,
        sources: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> dict[str, float]:
        assert len(sources) == len(
            targets
        ), f"Given sources ({len(sources)}) and targets ({len(targets)}) should contain same num of items."

        results = {}
        for metric_name, metric_fn in self.metrics.items():
            logger.info(f"Evaluate {metric_name}")
            result: float = metric_fn(sources, targets, **kwargs)
            results[metric_name] = result
        return results
