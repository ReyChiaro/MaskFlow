import math
import torch
import numpy as np
import dataclasses

from typing import Literal
from contextlib import contextmanager
from diffusers.training_utils import compute_loss_weighting_for_sd3

from schedulers.base_scheduler import BaseScheduler


@dataclasses.dataclass
class RectifiedFlowMatchingScheduler(BaseScheduler):

    weighting_scheme: Literal["logit_normal", "mode"] | None = ("logit_normal",)
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    mode_scale: float = 1.29
    base_image_seq_len: int = 256
    base_shift: float = 0.5
    max_image_seq_len: int = 8192
    max_shift: float = 0.9
    shift: float = 1.0
    shift_power: int = 1
    time_shift_type: Literal["exponential", "linear"] = "exponential"
    use_dynamic_shifting: bool = True

    def __post_init__(self):
        self.shift_mu = None if self.use_dynamic_shifting else self.shift

    def sample_timesteps(self, batch_size: int, generator=None, device=None) -> torch.Tensor:
        t = torch.randn((batch_size,), generator=generator, device=device, dtype=torch.float32)
        if self.weighting_scheme == "logit_normal":
            t = t * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(t)
        elif self.weighting_scheme == "mode":
            t = 1 - t - self.mode_scale * (torch.cos(math.pi * t / 2) ** 2 - 1 + t)
        return t

    def calculate_shift_mu(self, img_seq_len: int) -> float:
        scale = (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len)
        shift = self.base_shift - scale * self.base_image_seq_len
        return scale * img_seq_len + shift

    def time_shift(self, t: torch.Tensor, mu: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_shift_type == "exponential":
            sigmas = np.exp(mu) / (np.exp(mu) + (1.0 / t - 1) ** self.shift_power)
            loss_weights = (sigmas / t) ** 2 / np.exp(mu) * ((1.0 / t - 1) ** (self.shift_power - 1))
        elif self.time_shift_type == "linear":
            sigmas = mu / (mu + (1.0 / t - 1) ** self.shift_power)
            loss_weights = (sigmas / t) ** 2 / mu * ((1.0 / t - 1) ** (self.shift_power - 1))
        else:
            sigmas = t
            loss_weights = torch.ones_like(sigmas)
        return sigmas.to(t.device, dtype=t.dtype), loss_weights.to(t.device, dtype=t.dtype)

    def get_sigmas(
        self,
        t: torch.Tensor,
        img_seq_len: int | None = None,
        return_loss_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_dynamic_shifting:
            if img_seq_len is None:
                raise ValueError("img_seq_len must be provided when dynamic shifting is enabled.")
            mu = self.calculate_shift_mu(img_seq_len)
        else:
            mu = self.shift_mu
        sigmas, loss_weights = self.time_shift(t, mu)
        if return_loss_weights:
            return sigmas, loss_weights
        return sigmas

    def add_noise_by_sigmas(self, noise: torch.Tensor, x0: torch.Tensor, sigmas: torch.Tensor):
        device = x0.device
        dtype = x0.dtype
        while sigmas.ndim < x0.ndim:
            sigmas = sigmas.unsqueeze(-1)

        xt = (1.0 - sigmas) * x0.float() + sigmas * noise.float()
        return xt.to(device, dtype=dtype)

    def add_noise(self, noise: torch.Tensor, x0: torch.Tensor, t: torch.Tensor):
        # WARN: Deprecated
        sigmas = self.get_sigmas(t, img_seq_len=x0.shape[1], return_loss_weights=False)
        return self.add_noise_by_sigmas(noise, x0, sigmas)

    def get_velocity(self, noise: torch.Tensor, x0: torch.Tensor):
        return noise - x0

    def step(self, xt: torch.Tensor, vt: torch.Tensor, curr_sigma: torch.Tensor, next_sigma: torch.Tensor):
        dtype = xt.dtype
        return (xt.float() + (next_sigma - curr_sigma) * vt).to(dtype=dtype)

    def _inference(self, num_inference_steps: int, img_seq_len: int | None = None):
        timesteps = torch.from_numpy(
            np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, endpoint=True)
        ).float()
        sigmas = self.get_sigmas(timesteps, img_seq_len, return_loss_weights=False).float()
        sigmas = torch.cat([sigmas, torch.zeros((1,))])

        for step in range(num_inference_steps):
            curr_sigma = sigmas[step]
            next_sigma = sigmas[step + 1]
            yield sigmas[step : step + 1], curr_sigma, next_sigma

    @contextmanager
    def inference(self, num_inference_steps: int, img_seq_len: int | None = None):
        inference_it = self._inference(num_inference_steps, img_seq_len)
        try:
            yield inference_it
        finally:
            inference_it.close()
