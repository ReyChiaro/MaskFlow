import dataclasses

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

from .flow_matching import RectifiedFlowMatchingScheduler
from .mask_flow import MaskFlowScheduler


@dataclasses.dataclass
class Flux2FlowMatchingScheduler(RectifiedFlowMatchingScheduler):
    """Native FLUX.2 inference schedule with independently configured training sampling."""

    native_scheduler: FlowMatchEulerDiscreteScheduler | None = None
    weighting_scheme: str = "logit_normal"
    training_shift: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.training_shift <= 0:
            raise ValueError("training_shift must be positive.")

    def sample_timesteps(self, batch_size, generator=None, device=None):
        if self.weighting_scheme == "logit_normal":
            return super().sample_timesteps(batch_size, generator, device)
        t = torch.rand(batch_size, generator=generator, device=device)
        if self.weighting_scheme == "mode":
            return 1 - t - self.mode_scale * (torch.cos(torch.pi * t / 2).square() - 1 + t)
        if self.weighting_scheme in (None, "uniform"):
            return t
        raise ValueError(f"Unsupported weighting_scheme: {self.weighting_scheme}")

    def get_sigmas(self, t, img_seq_len=None, return_d_sigmas_dt=False):
        denominator = 1 + (self.training_shift - 1) * t
        sigmas = self.training_shift * t / denominator
        derivative = self.training_shift / denominator.square()
        return (sigmas, derivative) if return_d_sigmas_dt else sigmas

    def _inference(self, num_inference_steps, img_seq_len=None):
        if num_inference_steps <= 0 or img_seq_len is None:
            raise ValueError("Positive num_inference_steps and img_seq_len are required.")
        if self.native_scheduler is None:
            raise ValueError("The pipeline must load native_scheduler before inference.")
        # Use a fresh instance: evaluation must not change training or later evaluations.
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.native_scheduler.config)
        if scheduler.config.stochastic_sampling or scheduler.config.invert_sigmas:
            raise ValueError("FLUX.2 requires deterministic, non-inverted flow sigmas.")
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        if getattr(scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        scheduler.set_timesteps(
            num_inference_steps,
            sigmas=sigmas,
            mu=compute_empirical_mu(img_seq_len, num_inference_steps),
        )
        for index in range(num_inference_steps):
            sigma, next_sigma = scheduler.sigmas[index : index + 2]
            # The transformer and MaskFlow operate directly in sigma time, d(sigma)/d(sigma)=1.
            yield sigma.reshape(1), sigma, next_sigma, torch.ones_like(sigma)


@dataclasses.dataclass
class Flux2MaskFlowScheduler(Flux2FlowMatchingScheduler):
    """Compose the FLUX.2 time schedule with the existing MaskFlow dynamics."""

    unmask_with: str = "noisy_source"

    def __post_init__(self):
        super().__post_init__()
        if self.unmask_with not in {"source", "target", "noisy_source", "noisy_target"}:
            raise ValueError(f"Unsupported unmask_with: {self.unmask_with}")
        self._mask_scheduler = MaskFlowScheduler(unmask_with=self.unmask_with)

    def add_noise_by_sigmas(self, noise, x0, sigmas, source=None, mask=None):
        return self._mask_scheduler.add_noise_by_sigmas(noise, x0, sigmas, source, mask)

    def get_velocity(self, noise, x0, source=None, mask=None):
        return self._mask_scheduler.get_velocity(noise, x0, source, mask)

    def step(self, xt, vt, curr_sigma, next_sigma, d_sigma_dt, source=None, mask=None, noise=None):
        return self._mask_scheduler.step(xt, vt, curr_sigma, next_sigma, d_sigma_dt, source, mask, noise)

    def predict_x0(self, xt, sigma, d_sigma_dt, v, source=None, mask=None, noise=None):
        return self._mask_scheduler.predict_x0(xt, sigma, d_sigma_dt, v, source, mask, noise)
