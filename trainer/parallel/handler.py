import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from dataclasses import dataclass

from trainer.parallel.utils import get_world_size
from utils.singleton import singleton


@dataclass
@singleton
class ParallelHandler:

    global_mesh: DeviceMesh
    global_rank: int
    global_size: int

    dp_sp_mesh: DeviceMesh

    dp_mesh: DeviceMesh
    dp_group: dist.ProcessGroup
    dp_size: int
    dp_rank: int

    sp_mesh: DeviceMesh
    sp_group: dist.ProcessGroup
    sp_size: int
    sp_rank: int
    sp_group_idx: int

    def setup_parallel(self, sp_size: int = 1):

        if not dist.is_initialized():
            dist.init_process_group("nccl")
        world_size = get_world_size()
        assert world_size % sp_size == 0, f"world_size ({world_size}) should be divisible by sp_size ({sp_size})."

        dp_size = world_size // sp_size

        self.dp_sp_mesh = init_device_mesh("cuda", (dp_size, sp_size), mesh_dim_names=("DP", "SP"))

        self.dp_size = dp_size
        self.dp_mesh = self.dp_sp_mesh["DP"]
        self.dp_rank = self.dp_mesh.get_local_rank()
        self.dp_group = self.dp_sp_mesh["DP"].get_group()

        self.sp_size = sp_size
        self.sp_mesh = self.dp_sp_mesh["SP"]
        self.sp_rank = self.sp_mesh.get_local_rank()
        self.sp_group = self.dp_sp_mesh["SP"].get_group()

        self.global_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("default",))
        self.global_rank = self.global_mesh.get_local_rank()
        self.global_size = self.global_mesh.size()

        self.sp_group_idx = self.global_rank // self.sp_size

    def get_device_mesh(self) -> DeviceMesh:
        return self.global_mesh


parallel_handler = ParallelHandler()
