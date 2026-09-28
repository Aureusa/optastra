from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, IterableDataset

from .collate import CollateFn
from .sampler import ShardedSampler
from ..core.distributed import broadcast_object, env_world_size, get_rank, init_distributed

if TYPE_CHECKING:  # only for the annotation -- importing tasks here would create an import cycle
    from ..tasks import Task


def build_dataloader(
        dataset,
        task: "Task",
        batch_size: int,
        collate_kwargs: dict = None,
        *,
        distributed: bool | None = None,
        **kwargs,
    ) -> DataLoader:
    """
    The dataloader wires itself to whatever collate the task declares --
    the user never has to remember 'detection needs collate_fn=my_ragged_collate'.

    ``collate_kwargs`` are bound to the collate function, e.g.
    ``collate_kwargs={"size_divisibility": 32}`` for the ragged collate.

    Multi-process (DDP) runs: under torchrun (or with ``distributed=True``)
    each process gets a disjoint shard of ``dataset`` through a
    `ShardedSampler` -- ``shuffle=True`` reshuffles the whole dataset every
    epoch, and ``batch_size`` is *per process* (the global batch is
    ``batch_size * world_size``). Remaining ``kwargs`` go to the DataLoader.

    An ``IterableDataset`` has no indices to shard, so it must split its
    stream across processes itself (``optastra.core.distributed.get_rank()``
    / ``get_world_size()``, plus ``torch.utils.data.get_worker_info()`` for
    DataLoader workers). If it has a ``set_epoch(epoch)`` method, the Trainer
    calls it at the start of every epoch, like a sampler's.
    """
    collate_fn = CollateFn.create(task.collate, **(collate_kwargs or {}))
    if distributed is None:
        distributed = env_world_size() > 1
    shardable = not isinstance(dataset, IterableDataset) and "sampler" not in kwargs and "batch_sampler" not in kwargs
    if distributed and init_distributed() and shardable:
        # The shuffle seed must be identical on every process (or shards would
        # overlap), so everyone uses rank 0's torch seed.
        seed = broadcast_object(torch.initial_seed())
        kwargs["sampler"] = ShardedSampler(len(dataset), shuffle=kwargs.pop("shuffle", False), seed=seed)
        # DataLoader workers seed their RNGs from this generator: a different
        # stream per process, so each process augments differently.
        kwargs.setdefault("generator", torch.Generator().manual_seed((seed + get_rank()) % 2**63))
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, **kwargs)
