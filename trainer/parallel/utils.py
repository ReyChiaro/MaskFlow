import os
import torch
import functools

from loguru import logger
from typing import Callable, Any


def parallel_check(required_env: str | None = None, default_value: Any | None = None) -> Any:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            warnings = []

            if not torch.cuda.is_available():
                warnings.append("CUDA is not available.")

            if not torch.distributed.is_initialized() or not torch.distributed.is_available():
                warnings.append("Distributed not initialized or not available.")

            if required_env is not None and required_env not in os.environ:
                warnings.append(f"Environment variable not found: `{required_env}`. Use {default_value=} instead.")

            if warnings:
                logger.warning("\n" + "\n".join(warnings))
                return default_value
            return func(*args, **kwargs)

        return wrapper

    if callable(required_env):
        actual_func = required_env
        required_env = None
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


def is_distributed_usable() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def wait_for_everyone():
    if is_distributed_usable():
        torch.distributed.barrier()


def is_fsdp_module(module) -> bool:
    r"""
    For FSDP2, we only check if the module has attribute `set_requires_gradient_sync`
    """
    return hasattr(module, "set_requires_gradient_sync")
