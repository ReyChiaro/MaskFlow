import copy
import dataclasses
import math
from contextlib import contextmanager
from typing import Literal

import numpy as np
import torch

from .flow_matching import RectifiedFlowMatchingScheduler


@dataclasses.dataclass
class MaskFlowScheduler(RectifiedFlowMatchingScheduler):
    unmask_with: Literal["target", "source", "noisy_target", "noisy_source"] = "noisy_source"
    # noisy_source only: gamma >= 1; 1 keeps the foreground noise schedule.
    background_noise_power: float = 1.0

    def add_noise_by_sigmas(
        self,
        noise: torch.Tensor,
        x0: torch.Tensor,
        sigmas: torch.Tensor,
        source: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        if mask is None or source is None:
            return super().add_noise_by_sigmas(noise, x0, sigmas)
        device = x0.device
        dtype = x0.dtype

        while sigmas.ndim < x0.ndim:
            sigmas = sigmas.unsqueeze(-1)

        x0 = x0.float()
        noise = noise.float()

        if self.unmask_with == "target":
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * x0
        elif self.unmask_with == "source":
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * source
        elif self.unmask_with == "noisy_target":
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * ((1.0 - sigmas) * x0 + sigmas * noise)
        elif self.unmask_with == "noisy_source":
            background_sigmas = sigmas**self.background_noise_power
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * (
                (1.0 - background_sigmas) * source + background_sigmas * noise
            )
        else:
            raise KeyError(
                f"{self.unmask_with=} is not supported. Acceptable values are [target, source, noisy_target, noisy_source]."
            )

        return xt.to(device, dtype=dtype)

    def add_noise(
        self,
        noise,
        x0,
        t,
        source: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        sigmas = self.get_sigmas(t, img_seq_len=x0.shape[1], return_d_sigmas_dt=False)
        return self.add_noise_by_sigmas(noise, x0, sigmas, source, mask)

    def get_velocity(
        self,
        noise: torch.Tensor,
        x0: torch.Tensor,
        source: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        sigmas: torch.Tensor | None = None,
    ):
        if source is None or mask is None:
            return super().get_velocity(noise, x0)

        if self.unmask_with == "target":
            return (mask * noise + (1 - mask) * x0) - (mask * x0 + (1 - mask) * x0)
        elif self.unmask_with == "source":
            return (mask * noise + (1 - mask) * source) - (mask * x0 + (1 - mask) * source)
        elif self.unmask_with == "noisy_target":
            return (mask * noise + (1 - mask) * noise) - (mask * x0 + (1 - mask) * x0)
        elif self.unmask_with == "noisy_source":
            if self.background_noise_power != 1.0:
                # Required for gamma != 1: shifted sigma, scalar or [batch].
                while sigmas.ndim < x0.ndim:
                    sigmas = sigmas.unsqueeze(-1)
                background_rate = self.background_noise_power * sigmas ** (self.background_noise_power - 1.0)
                # Velocity is dx/dsigma, not dx/dt before time_shift.
                return mask * (noise - x0) + (1 - mask) * background_rate * (noise - source)
            return (mask * noise + (1 - mask) * noise) - (mask * x0 + (1 - mask) * source)
        else:
            raise KeyError(
                f"{self.unmask_with=} is not supported. Acceptable values are [target, source, noisy_target, noisy_source]."
            )

    def step(
        self,
        xt,
        vt,
        curr_sigma,
        next_sigma,
        d_sigma_dt,
        source: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ):
        if source is None or mask is None or noise is None:
            return super().step(xt, vt, curr_sigma, next_sigma, d_sigma_dt)

        xt = super().step(xt, vt, curr_sigma, next_sigma, d_sigma_dt)

        if self.unmask_with in ["source", "target"]:
            xt = mask * xt + (1 - mask) * source
        elif self.unmask_with == "noisy_source":
            background_sigma = next_sigma**self.background_noise_power
            xt = mask * xt + (1 - mask) * ((1.0 - background_sigma) * source + background_sigma * noise)
        elif self.unmask_with == "noisy_target":
            xt = mask * xt + (1 - mask) * ((1.0 - next_sigma) * source + next_sigma * noise)

        return xt

    def predict_x0(
        self,
        xt: torch.Tensor,
        sigma: torch.Tensor,
        d_sigma_dt: torch.Tensor,
        v: torch.Tensor,
        source: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ):
        dtype = xt.dtype
        while sigma.ndim < xt.ndim:
            sigma = sigma.unsqueeze(-1)
        x0 = xt.float() - sigma.float() * v.float()
        if self.unmask_with == "noisy_source" and self.background_noise_power != 1.0:
            # Source, mask and the path's original noise are required for this correction.
            correction = (self.background_noise_power - 1.0) * (sigma.float() ** self.background_noise_power)
            x0 = x0 + (1 - mask.float()) * correction * (noise.float() - source.float())
        return x0.to(dtype=dtype)
