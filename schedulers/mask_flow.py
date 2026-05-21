import copy
import math
import torch
import numpy as np
import dataclasses

from typing import Literal
from contextlib import contextmanager

from .flow_matching import RectifiedFlowMatchingScheduler


@dataclasses.dataclass
class MaskFlowScheduler(RectifiedFlowMatchingScheduler):

    unmask_with: Literal["target", "source", "noisy_target", "noisy_source"] = "noisy_source"

    def add_noise(self, noise, x0, t, source: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        if mask is None or source is None:
            return super().add_noise(noise, x0, t)
        device = x0.device
        dtype = x0.dtype
        if self.use_dynamic_shifting:
            img_seq_len = x0.shape[1]
            mu = self.calculate_shift_mu(img_seq_len)
        else:
            mu = self.shift_mu
        sigmas = self.time_shift(t, mu)  # [B,], float32

        x0 = x0.float()
        noise = noise.float()

        if self.unmask_with == "target":
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * x0
        elif self.unmask_with == "source":
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * source
        elif self.unmask_with == "noisy_target":
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * ((1.0 - sigmas) * x0 + sigmas * noise)
        elif self.unmask_with == "noisy_source":
            xt = mask * ((1.0 - sigmas) * x0 + sigmas * noise) + (1 - mask) * ((1.0 - sigmas) * source + sigmas * noise)
        else:
            raise KeyError(
                f"{self.unmask_with=} is not supported. Acceptable values are [target, source, noisy_target, noisy_source]."
            )

        return xt.to(device, dtype=dtype), sigmas.to(device, dtype=dtype)

    def get_velocity(self, noise, x0, source: torch.Tensor | None = None, mask: torch.Tensor | None = None):
        if source is None or mask is None:
            return super().get_velocity(noise, x0)

        if self.unmask_with == "target":
            return (mask * noise + (1 - mask) * x0) - (mask * x0 + (1 - mask) * x0)
        elif self.unmask_with == "source":
            return (mask * noise + (1 - mask) * source) - (mask * x0 + (1 - mask) * source)
        elif self.unmask_with == "noisy_target":
            return (mask * noise + (1 - mask) * noise) - (mask * x0 + (1 - mask) * x0)
        elif self.unmask_with == "noisy_source":
            return (mask * noise + (1 - mask) * noise) - (mask * x0 + (1 - mask) * source)
        else:
            raise KeyError(
                f"{self.unmask_with=} is not supported. Acceptable values are [target, source, noisy_target, noisy_source]."
            )

    def _inference(
        self,
        xt: torch.Tensor,
        num_inference_steps: int,
        source: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        img_seq_len: int | None = None,
    ):
        @dataclasses.dataclass
        class _Inferencer:
            pred_v: torch.Tensor = dataclasses.field(init=None)

            def step(self, v):
                self.pred_v = v

        if self.use_dynamic_shifting:
            mu = self.calculate_shift_mu(img_seq_len)
        else:
            mu = self.shift_mu

        noise = copy.deepcopy(xt)
        timesteps = torch.from_numpy(
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps, endpoint=True)
        ).float()
        sigmas = self.time_shift(timesteps, mu).float()
        sigmas = torch.cat([sigmas, torch.zeros((1,))])
        for step in range(num_inference_steps + 1):
            inferencer = _Inferencer()
            try:
                yield xt, sigmas[step : step + 1], inferencer
            finally:
                if step >= num_inference_steps:
                    continue
                sigma = sigmas[step]
                sigma_next = sigmas[step + 1]
                xt = self.step(xt.float(), inferencer.pred_v, sigma_next - sigma).to(xt.device, dtype=xt.dtype)

                # Handle masks
                if source is not None and mask is not None:
                    if self.unmask_with in ["source", "target"]:
                        xt = mask * xt + (1 - mask) * source
                    elif self.unmask_with in ["noisy_source", "noisy_target"]:
                        xt = mask * xt + (1 - mask) * ((1.0 - sigma_next) * source + sigma_next * noise)

    @contextmanager
    def inference_sampler(
        self,
        xt: torch.Tensor,
        num_inference_steps: int,
        source: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        img_seq_len: int | None = None,
    ):
        inference_it = self._inference(xt, num_inference_steps, source, mask, img_seq_len)
        try:
            yield inference_it
        finally:
            inference_it.close()
