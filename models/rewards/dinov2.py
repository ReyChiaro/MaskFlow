from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from models.rewards.rewards import RewardModel


@dataclass
class DINOv2Reward(RewardModel):
    r"""Per-image CLS cosine similarity to source or target, chosen explicitly.

    source measures visual preservation; target needs a dataset target image.
    Each image is resized by the pretrained processor independently.
    """

    pretrained_model: str = ""
    batch_size: int = 8
    reference: Literal["source", "target"] = "source"

    def __post_init__(self):
        if not self.pretrained_model or self.batch_size < 1 or self.reference not in ("source", "target"):
            raise ValueError("DINOv2 requires pretrained_model, positive batch_size and reference=source/target.")
        self.processor = AutoImageProcessor.from_pretrained(self.pretrained_model)
        self.model = AutoModel.from_pretrained(self.pretrained_model).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        reference = batch["conditions"]["source"] if self.reference == "source" else batch.get("target")
        if reference is None or len(reference) != len(images):
            raise ValueError(f"DINOv2 requires one {self.reference} image per generated image.")
        scores = []
        for start in range(0, len(images), self.batch_size):
            count = min(self.batch_size, len(images) - start)
            pairs = list(images[start : start + count].detach().cpu().float())
            pairs.extend(reference[start : start + count].detach().cpu().float())
            inputs = self.processor(images=pairs, return_tensors="pt", do_rescale=False).to(self.device)
            features = F.normalize(self.model(**inputs).last_hidden_state[:, 0].float(), dim=-1)
            scores.append((features[:count] * features[count:]).sum(-1))
        return torch.cat(scores).to(images.device)
