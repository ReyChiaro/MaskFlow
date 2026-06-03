from enum import StrEnum


class FSDPStrategy(StrEnum):

    FULL_SHARD = "full_shard"
    NO_SHARD = "no_shard"

    @staticmethod
    def is_no_shard(strategy: "FSDPStrategy|str") -> bool:
        return strategy == FSDPStrategy.NO_SHARD

    @staticmethod
    def is_full_shard(strategy: "FSDPStrategy|str") -> bool:
        return strategy == FSDPStrategy.FULL_SHARD
