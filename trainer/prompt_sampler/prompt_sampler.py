import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Type


class ProbabilitySchedule:
    def __call__(self, step: int) -> float:
        raise NotImplementedError


SCHEDULER_REGISTRY: dict[str, Type[ProbabilitySchedule]] = {}


def register_scheduler(name: Optional[str] = None, override: bool = False):

    def decorator(cls: Type[ProbabilitySchedule]):
        if not issubclass(cls, ProbabilitySchedule):
            raise TypeError(f"{cls.__name__} must inherit from ProbabilitySchedule.")

        register_names = [name or cls.__name__]
        for scheduler_name in register_names:
            scheduler_name = scheduler_name.lower()
            if scheduler_name in SCHEDULER_REGISTRY and not override:
                raise ValueError(f"Scheduler '{scheduler_name}' is already registered.")
            SCHEDULER_REGISTRY[scheduler_name] = cls

        return cls

    return decorator


@register_scheduler("constant")
class ConstantSchedule(ProbabilitySchedule):

    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__()
        self.p: float = p

    def __call__(self, step: int) -> float:
        return self.p


@register_scheduler("linear-decay")
class LinearDecaySchedule(ProbabilitySchedule):

    def __init__(self, start_p: float = 1.0, end_p: float = 0.0, total_steps: int = 10000, **kwargs):
        super().__init__()
        self.start_p = start_p
        self.end_p = end_p
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        ratio = min(max(step / self.total_steps, 0.0), 1.0)
        return self.start_p + ratio * (self.end_p - self.start_p)


@register_scheduler("const-linear")
class ConstantLinearDecaySchedule(ProbabilitySchedule):

    def __init__(self, start_p: float, end_p: float, decay_start_step: int, decay_end_step: int, **kwargs):
        super().__init__()
        self.start_p = start_p
        self.end_p = end_p
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step

    def __call__(self, step: int) -> float:
        if step < self.decay_start_step:
            return self.start_p

        if step >= self.decay_end_step:
            return self.end_p

        ratio = (step - self.decay_start_step) / (self.decay_end_step - self.decay_start_step)
        return self.start_p + ratio * (self.end_p - self.start_p)


class PromptSampler:

    def __init__(self, name: str, seed: int = 42, **kwargs):
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.schedule = SCHEDULER_REGISTRY[name](**kwargs)

    def sample_one(self, prompt_0: str, prompt_1: str, step: int) -> str:
        p = self.schedule(step)
        p = min(max(p, 0.0), 1.0)
        return prompt_0 if self.rng.random() < p else prompt_1

    def sample_batch(self, prompt_pairs: Sequence[tuple[str, str]], step: int) -> list[str]:
        return [self.sample_one(prompt_0, prompt_1, step) for prompt_0, prompt_1 in prompt_pairs]
