from enum import StrEnum


class FSDPStrategy(StrEnum):

    FULL_SHARD = "full_shard"
    NO_SHARD = "no_shard"
