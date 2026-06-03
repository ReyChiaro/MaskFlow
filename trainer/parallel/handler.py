import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from dataclasses import dataclass

from trainer.parallel.utils import get_world_size
from utils.singleton import singleton


@dataclass
@singleton
class ParallelHandler:
    r"""
    Data Parallel handler.
    """

    global_mesh: DeviceMesh
    global_rank: int
    global_size: int

    dp_mesh: DeviceMesh
    dp_group: dist.ProcessGroup
    dp_size: int
    dp_rank: int

    def setup_parallel(self):
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        world_size = get_world_size()

        self.global_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("DP",))
        self.global_rank = self.global_mesh.get_local_rank()
        self.global_size = self.global_mesh.size()

        self.dp_mesh = self.global_mesh
        self.dp_group = self.dp_mesh.get_group()
        self.dp_size = self.dp_mesh.size()
        self.dp_rank = self.dp_mesh.get_local_rank()

    def get_device_mesh(self) -> DeviceMesh:
        return self.dp_mesh


parallel_handler = ParallelHandler()
