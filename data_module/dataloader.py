from torch.utils.data import Dataset, DataLoader

from data_module.sampler import CheckpointDistributedSampler


def get_dataloader(
    dataset: Dataset,
    batch_size_per_process: int,
    num_workers: int,
    num_replicas: int = 1,
    global_rank: int = 0,
    global_seed: int = 0,
    drop_last: bool = False,
    is_train: bool = True,
):
    sampler = None
    if num_replicas > 1:
        sampler = CheckpointDistributedSampler(
            dataset=dataset,
            batch_size=batch_size_per_process,
            num_replicas=num_replicas,
            rank=global_rank,
            shuffle=is_train,
            seed=global_seed,
            drop_last=drop_last,
        )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size_per_process,
        shuffle=(sampler is None or is_train),
        num_workers=num_workers,
        drop_last=drop_last,
        sampler=sampler,
    )
    return dataloader, sampler
