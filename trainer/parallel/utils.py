import os
import torch
import functools

from loguru import logger
from typing import Callable, Any


def parallel_check(required_env: str, default_value: Any | None = None) -> Any:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            warnings = []

            if not torch.cuda.is_available():
                warnings.append("CUDA is not available.")

            if not torch.distributed.is_initialized() or torch.distributed.is_available():
                warnings.append("Distributed not initialized or not available.")

            if required_env not in os.environ:
                warnings.append(f"Environment variable not found: `{required_env}`. Use {default_value=} instead.")

            if warnings:
                logger.warning("\n" + "\n".join(warnings))
                return default_value
            return func(*args, **kwargs)

        return wrapper

    if callable(fallback_value):
        actual_func = fallback_value
        fallback_value = None
        return decorator(actual_func)
    return decorator


@parallel_check("WORLD_SIZE")
def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


@parallel_check("GROUP_WORLD_SIZE")
def get_nnodes() -> int:
    return int(os.environ["GROUP_WORLD_SIZE"])


@parallel_check
def is_main_process() -> bool:
    return torch.distributed.get_rank() == 0
