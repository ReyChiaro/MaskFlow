from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


@dataclass
class RewardModel(ABC):
    r"""Per-image reward interface. Inputs are RGB [B,C,H,W] in [0,1].

    batch contains the corresponding dataset fields (prompt, conditions,
    optional target). Return finite float scores [B]; higher is always better.
    Models own their frozen weights; there is no process-global model cache.
    """

    device: str | torch.device = "cpu"

    @abstractmethod
    def __call__(self, images: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class Rewards(RewardModel):
    r"""Hydra-configured weighted sum of RewardModel instances, without averaging B."""

    models: OmegaConf | None = None

    def __post_init__(self):
        if not self.models:
            raise ValueError("At least one reward model must be configured.")
        self.reward_models: dict[str, RewardModel] = {}
        self.weights: dict[str, float] = {}
        for name, config in self.models.items():
            weight = float(config.weight)
            if not isfinite(weight):
                raise ValueError(f"Reward weight for {name} must be finite.")
            if weight == 0:
                continue
            model = instantiate(config.model, device=self.device)
            if not isinstance(model, RewardModel):
                raise TypeError(f"Reward {name} must implement RewardModel.")
            self.reward_models[name] = model
            self.weights[name] = weight
        if not self.reward_models:
            raise ValueError("At least one reward weight must be nonzero.")

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        result = torch.zeros(len(images), device=images.device, dtype=torch.float32)
        for name, model in self.reward_models.items():
            scores = torch.as_tensor(model(images=images, batch=batch), device=images.device, dtype=torch.float32)
            if scores.shape != result.shape or not torch.isfinite(scores).all():
                raise ValueError(f"Reward {name} must return finite scores with shape [{len(images)}].")
            result.add_(scores, alpha=self.weights[name])
        return result
