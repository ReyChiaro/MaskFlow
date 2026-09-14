from dataclasses import dataclass
from typing import Any

import torch
from transformers import CLIPModel, CLIPProcessor

from models.rewards.rewards import RewardModel


@dataclass
class CLIPReward(RewardModel):
    r"""Cosine similarity of the edited image and batch[prompt_key], per image."""

    pretrained_model: str = ""
    batch_size: int = 8
    prompt_key: str = "prompt"

    def __post_init__(self):
        if not self.pretrained_model or self.batch_size < 1:
            raise ValueError("CLIP requires pretrained_model and a positive batch_size.")
        self.processor = CLIPProcessor.from_pretrained(self.pretrained_model)
        self.model = CLIPModel.from_pretrained(self.pretrained_model).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        prompts = batch[self.prompt_key]
        if len(prompts) != len(images) or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("CLIP requires one prompt string per image.")
        scores = []
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(
                text=list(prompts[start : start + self.batch_size]),
                images=list(images[start : start + self.batch_size].detach().cpu().float()),
                return_tensors="pt",
                padding=True,
                truncation=True,
                do_rescale=False,
                max_length=self.model.config.text_config.max_position_embeddings,
            ).to(self.device)
            output = self.model(**inputs)
            scores.append((output.image_embeds.float() * output.text_embeds.float()).sum(-1))
        return torch.cat(scores).to(images.device)
