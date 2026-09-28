from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler

from ..core.distributed import get_rank, get_world_size


__all__ = ["ShardedSampler"]


class ShardedSampler(Sampler[int]):
    """
    Gives each process a disjoint share of a dataset for multi-process (DDP)
    training and evaluation: process `rank` reads indices
    `order[rank], order[rank + world_size], ...`.

    Unlike torch's DistributedSampler it neither pads nor drops samples, so
    every sample is seen exactly once per epoch across all processes --
    evaluation metrics stay exact, at the cost of shards differing in length
    by at most one sample (harmless for the iteration-based Trainer).

    With `shuffle=True`, `order` is a fresh permutation every epoch
    (`set_epoch`, which the Trainer calls for you), drawn from `seed + epoch`
    -- the same on every process, which is what keeps the shards disjoint.
    """

    def __init__(
            self,
            num_samples: int,
            shuffle: bool = False,
            seed: int = 0,
            rank: int | None = None,
            world_size: int | None = None,
        ):
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.rank = get_rank() if rank is None else rank
        self.world_size = get_world_size() if world_size is None else world_size
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is outside [0, {self.world_size}).")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator().manual_seed((self.seed + self.epoch) % 2**63)
            order = torch.randperm(self.num_samples, generator=generator)
        else:
            order = torch.arange(self.num_samples)
        return iter(order[self.rank::self.world_size].tolist())

    def __len__(self) -> int:
        return len(range(self.rank, self.num_samples, self.world_size))
