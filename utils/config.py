import os
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

_distributed_timestamps: dict[str, str] = {}


def distributed_timestamp(date_format: str = "%Y%m%d-%H%M%S") -> str:
    r"""Generate one timestamp on rank 0 and share it with every training rank."""
    if date_format in _distributed_timestamps:
        return _distributed_timestamps[date_format]

    timestamp = datetime.now().strftime(date_format)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed timestamp synchronization requires CUDA for the NCCL training group.")

        global_rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", global_rank % torch.cuda.device_count()))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)

        if not dist.is_initialized():
            timeout_seconds = int(os.environ.get("MASKFLOW_DISTRIBUTED_TIMEOUT_SECONDS", 21600))
            dist.init_process_group(
                "nccl",
                device_id=device,
                timeout=timedelta(seconds=timeout_seconds),
            )

        timestamp_list = [timestamp if global_rank == 0 else ""]
        dist.broadcast_object_list(timestamp_list, src=0, device=device)
        timestamp = timestamp_list[0]

    _distributed_timestamps[date_format] = timestamp
    return timestamp


def register_distributed_timestamp_resolver():
    OmegaConf.register_new_resolver(
        "distributed_timestamp",
        distributed_timestamp,
        replace=True,
        use_cache=True,
    )
