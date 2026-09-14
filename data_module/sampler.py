from collections.abc import Iterator

from loguru import logger
from torch.utils.data import Dataset, DistributedSampler


class CheckpointDistributedSampler(DistributedSampler):
    r"""
    Distributed dataset sampler supporting recover from training epoch and its step
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int | None = 0,
        drop_last: bool | None = False,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.batch_size: int = batch_size

        # The actual data sample idx
        self._resume_idx: int = 0

        # Used for iter start
        self.resume_idx: int = 0

    def reset(self):
        self._resume_idx = 0

    def set_resume_idx(self, resume_idx):
        self.resume_idx = resume_idx

    def set_epoch(self, epoch):
        r"""
        Set epoch and reset the resume idx used for iter loop to zero.
        """
        self.epoch = epoch
        self.set_resume_idx(0)

    def __len__(self) -> int:
        return self.num_samples - self.resume_idx

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)

        for i in range(self.resume_idx, len(indices)):
            self._resume_idx = i + 1
            idx = indices[i]
            yield idx

    def state_dict(self) -> dict:
        if self._resume_idx == self.num_samples:
            self._resume_idx = 0
            self.epoch = 0
        return {
            "resume_idx": self._resume_idx,
            "epoch": self.epoch,
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state_dict: dict):
        if self.seed != state_dict["seed"]:
            logger.warning(
                f"Random seed mismatch: state_dict: {state_dict['seed']}, current: {self.seed}. This may influence the random strategy in dataset sampling."
            )
        if self.num_replicas != state_dict["num_replicas"]:
            logger.warning(
                f"Number of replicas mismatch: state_dict: {state_dict['num_replicas']}, current: {self.num_replicas}. This may generate uncontinuous data samples."
            )
        self.set_epoch(state_dict["epoch"])
        self.set_resume_idx(state_dict["resume_idx"])
