"""
Seedable randomness for transforms.

Every random decision a transform makes (flip or not, which op, which
magnitude, where to crop, ...) is drawn from ONE `torch.Generator` per
process, returned by `get_generator()`. Transforms fetch it at call time,
never at construction time, so the same Transform object behaves correctly
after being copied into DataLoader worker processes.

Seeding rules, in order of precedence:

1. `seed_transforms(seed)` -- explicit seed for the current process.
2. Otherwise the generator is seeded from `torch.initial_seed()` the first
   time it is used in a process. So `torch.manual_seed(s)` at the top of a
   script also makes transforms reproducible.
3. DataLoader workers: a forked worker inherits the parent's generator, so
   without care every worker would produce the *same* augmentations. The
   generator therefore remembers which process created it and re-seeds
   itself on first use in a new process from `torch.initial_seed()`, which
   PyTorch sets to `base_seed + worker_id` inside each worker. Seed the
   DataLoader (`generator=torch.Generator().manual_seed(s)`) or call
   `torch.manual_seed(s)` before building it and each worker's stream is
   reproducible. `worker_init_fn` does the same thing explicitly, for those
   who prefer to see it in their DataLoader call.

The small draw helpers below (`uniform`, `choice`, ...) mirror the
`random` module API so transform code stays readable.
"""
from __future__ import annotations

import os
from typing import Sequence, TypeVar

import numpy as np
import torch


__all__ = [
    "get_generator", "seed_transforms", "worker_init_fn",
    "rand", "uniform", "randint", "choice", "sample", "sign", "randperm", "beta", "dirichlet",
]

T = TypeVar("T")

_generator: torch.Generator | None = None
_owner_pid: int | None = None


def get_generator() -> torch.Generator:
    """The CPU generator every transform draws from in this process."""
    global _generator, _owner_pid
    if _generator is None or _owner_pid != os.getpid():
        _generator = torch.Generator()
        _generator.manual_seed(torch.initial_seed())
        _owner_pid = os.getpid()
    return _generator


def seed_transforms(seed: int) -> None:
    """Re-seed the transform generator of the current process."""
    get_generator().manual_seed(seed)


def worker_init_fn(worker_id: int) -> None:
    """`DataLoader(worker_init_fn=...)`-compatible: seeds this worker's
    transform generator from `torch.initial_seed()` (= base_seed + worker_id)."""
    seed_transforms(torch.initial_seed())


# ---- draw helpers ---------------------------------------------------------

def rand() -> float:
    """Uniform float in [0, 1)."""
    return float(torch.rand((), generator=get_generator()))


def uniform(low: float, high: float) -> float:
    """Uniform float in [low, high)."""
    return low + (high - low) * rand()


def randint(low: int, high: int) -> int:
    """Uniform integer in [low, high] -- both ends inclusive, like `random.randint`."""
    return int(torch.randint(low, high + 1, (), generator=get_generator()))


def choice(seq: Sequence[T]) -> T:
    """One element of a non-empty sequence."""
    return seq[randint(0, len(seq) - 1)]


def sample(seq: Sequence[T], k: int) -> list[T]:
    """`k` distinct elements, in random order (sampling without replacement)."""
    if k > len(seq):
        raise ValueError(f"Cannot sample {k} elements from a sequence of length {len(seq)}.")
    return [seq[i] for i in randperm(len(seq))[:k].tolist()]


def sign() -> int:
    """-1 or +1 with equal probability."""
    return choice((-1, 1))


def randperm(n: int) -> torch.Tensor:
    """Random permutation of 0..n-1 (on CPU; move it to your device yourself)."""
    return torch.randperm(n, generator=get_generator())


def _numpy_rng() -> np.random.Generator:
    # torch has no generator-aware Beta/Dirichlet sampler, so derive a
    # one-off numpy generator from our stream -- still fully seeded.
    return np.random.default_rng(int(torch.randint(2**62, (), generator=get_generator())))


def beta(a: float, b: float) -> float:
    """One draw from Beta(a, b)."""
    return float(_numpy_rng().beta(a, b))


def dirichlet(alpha: float, n: int) -> torch.Tensor:
    """One draw from a symmetric Dirichlet(alpha) over `n` categories, as float32."""
    return torch.from_numpy(_numpy_rng().dirichlet([alpha] * n)).to(torch.float32)
