"""
Multi-process (DistributedDataParallel) support: one process per GPU.

Launch the *same* training script with torchrun instead of python:

    python   train.py                          # 1 process, as always
    torchrun --nproc_per_node=2 train.py       # 2 processes, e.g. 2 GPUs on one node

torchrun sets RANK / WORLD_SIZE / LOCAL_RANK (and MASTER_ADDR / MASTER_PORT)
in each process's environment; `Trainer` and `build_dataloader` read them and
set everything up. Under plain `python` every helper below returns the
single-process answer (rank 0 of 1), so code can call them unconditionally.

On a SLURM cluster, start torchrun from the job step, e.g.
`srun torchrun --standalone --nproc_per_node=2 train.py` for one node.
"""
from __future__ import annotations

import datetime
import os
from typing import Any

import torch
import torch.distributed as dist


__all__ = [
    "env_world_size", "init_distributed", "cleanup",
    "is_distributed", "get_rank", "get_local_rank", "get_world_size", "is_main_process",
    "barrier", "all_reduce_mean", "broadcast_object", "sum_across_processes",
]


def env_world_size() -> int:
    """Number of processes the launcher started (torchrun's WORLD_SIZE), 1 if none."""
    return int(os.environ.get("WORLD_SIZE", 1))


def init_distributed(backend: str | None = None, timeout: datetime.timedelta | None = None) -> bool:
    """
    Join the process group the launcher set up. Idempotent: returns True if
    this process is (now) part of a multi-process run, False when running as
    a single process -- in which case nothing is done.

    Called automatically by `Trainer` and `build_dataloader` under torchrun;
    call it yourself first only to pick the backend or timeout.

    :param backend: None = "cpu:gloo,cuda:nccl" when CUDA is available (NCCL
        for GPU tensors, gloo for CPU ones), else "gloo".
    """
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return dist.get_world_size() > 1
    if env_world_size() <= 1:
        return False

    if backend is None:
        backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available() and get_local_rank() < torch.cuda.device_count():
        torch.cuda.set_device(get_local_rank())   # NCCL collectives use the current device
    dist.init_process_group(backend=backend, **({"timeout": timeout} if timeout else {}))

    # Different augmentation randomness per process: the transforms' generator
    # is otherwise seeded from torch.initial_seed(), which is usually the same
    # on every rank (same torch.manual_seed at the top of the script).
    from ..transforms.rng import seed_transforms
    seed_transforms((torch.initial_seed() + get_rank()) % 2**63)
    return True


def cleanup() -> None:
    """Leave the process group (call once at the end of a torchrun script)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Index of this process on its node -- i.e. which GPU it should use."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """True on rank 0 (and in single-process runs): the one process that
    writes checkpoints and logs."""
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Mean of `tensor` over all processes (a copy; the input is untouched)."""
    if not is_distributed():
        return tensor
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor)
    return tensor / get_world_size()


def broadcast_object(obj: Any) -> Any:
    """Rank 0's `obj`, on every process (any picklable object)."""
    if not is_distributed():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def sum_across_processes(values: dict[str, Any]) -> dict[str, Any]:
    """
    Per-key sum of `values` over all processes. Values are numbers or tensors
    (tensors are returned on the CPU); a key only needs to exist on some
    processes -- e.g. a rank that saw no eval batches can send an empty dict.

    This is how Evaluators combine their accumulators (counts, running sums)
    after a sharded evaluation pass.
    """
    if not is_distributed():
        return dict(values)
    local = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in values.items()}
    gathered: list[dict[str, Any]] = [None] * get_world_size()   # type: ignore[list-item]
    dist.all_gather_object(gathered, local)
    total: dict[str, Any] = {}
    for part in gathered:
        for k, v in part.items():
            total[k] = v if k not in total else total[k] + v
    return total
