import math
import torch
import numpy as np

from typing import Literal, Iterator


class RectifiedFlowMatchingScheduler:

    def __init__(
        self,
        weighting_scheme: Literal["logit_normal", "mode"] | None = "logit_normal",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        mode_scale: float = 1.29,
        base_image_seq_len: int = 256,
        base_shift: float = 0.5,
        max_image_seq_len: int = 8192,
        max_shift: float = 0.9,
        shift: float = 1.0,
        shift_power: int = 1,
        time_shift_type: Literal["exponential", "linear"] = "exponential",
        use_dynamic_shifting: bool = True,
    ):
        self.weighting_scheme = weighting_scheme
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.mode_scale = mode_scale
        self.time_shift_type = time_shift_type

        self.use_dynamic_shift = use_dynamic_shifting
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.shift_mu = None if self.use_dynamic_shift else shift
        self.shift_power = shift_power

    def sample_timesteps(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        t = torch.randn((batch_size,), generator=generator, device=device, dtype=dtype)
        if self.weighting_scheme == "logit_normal":
            t = t * self.logit_normal_std + self.logit_normal_mean
            t = torch.sigmoid(t)
        elif self.weighting_scheme == "mode":
            t = 1 - t - self.mode_scale * (torch.cos(math.pi * t / 2) ** 2 - 1 + t)
        return t

    def get_inference_sigmas(
        self,
        num_inference_steps: int,
        mu: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        mu = mu if mu is not None else self.shift_mu
        t = torch.linspace(1, 0, num_inference_steps + 1, device=device, dtype=dtype)[:-1]
        sigmas = self.time_shift(t, mu)
        return sigmas

    def inference_delta_sigmas(
        self, num_inference_steps: int, mu: float | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        mu = mu if mu is not None else self.shift_mu
        ts = torch.tensor(list(range(num_inference_steps, -1, -1))) / num_inference_steps
        sigmas = self.time_shift(ts, mu)
        print(f"sigmas: {sigmas}")
        for step in range(num_inference_steps):
            delta = sigmas[step + 1] - sigmas[step]
            yield delta, sigmas[step : step + 1]

    def calculate_shift_mu(self, img_seq_len: int) -> float:
        scale = (self.max_shift - self.base_shift) / (self.max_image_seq_len - self.base_image_seq_len)
        shift = self.base_shift - scale * self.base_image_seq_len
        return scale * img_seq_len + shift

    def time_shift(self, t: torch.Tensor, mu: float) -> torch.Tensor:
        if self.time_shift_type == "exponential":
            sigmas = np.exp(mu) / (np.exp(mu) + (1.0 / t - 1) ** self.shift_power)
        elif self.time_shift_type == "linear":
            sigmas = mu / (mu + (1.0 / t - 1) ** self.shift_power)
        else:
            sigmas = t
        return sigmas.to(t.device, dtype=t.dtype)

    def add_noise[T](self, sigmas: T, noise: T, latents: T) -> T:
        return (1.0 - sigmas) * latents + sigmas * noise

    def get_ground_truth[T](self, noise: T, latents: T) -> T:
        return noise - latents
