import math
from dataclasses import dataclass
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from .scheduler_base import Scheduler


__all__ = ["WarmupCosine", "WarmupCosineConfig", "warmup_cosine"]


@dataclass
class WarmupCosineConfig:
    # Required: the length of the whole schedule, usually the Trainer's
    # max_iter (in optimizer steps). There is deliberately no default -- a
    # silent default shorter than the run leaves the LR at its floor for the
    # rest of training. None here only means "not set yet" so the registered
    # default config can exist; WarmupCosine raises if it is still None.
    total_steps: int | None = None
    warmup_steps: int = 500
    warmup_start_factor: float = 0.01   # warmup begins at this fraction of base_lr
    min_lr_factor: float = 0.0          # cosine floor, as a fraction of base_lr


class WarmupCosine(lr_scheduler.LambdaLR):
    """Linear warmup from `warmup_start_factor * base_lr` to `base_lr` over
    `warmup_steps`, then cosine decay to `min_lr_factor * base_lr` at
    `total_steps` (and flat after that)."""

    def __init__(self, optimizer: optim.Optimizer, cfg: WarmupCosineConfig):
        if cfg.total_steps is None:
            raise ValueError(
                "warmup_cosine requires `total_steps` (usually the Trainer's max_iter), e.g. "
                "Scheduler.create('warmup_cosine', optimizer, total_steps=max_iter)."
            )
        if cfg.total_steps < cfg.warmup_steps:
            raise ValueError(
                f"warmup_cosine: total_steps ({cfg.total_steps}) must be >= warmup_steps ({cfg.warmup_steps})."
            )
        self.cfg = cfg
        super().__init__(optimizer, lr_lambda=self._warmup_cosine_lambda(cfg))

    @staticmethod
    def _warmup_cosine_lambda(cfg: WarmupCosineConfig):
        def fn(step: int) -> float:
            if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
                # linear warmup: warmup_start_factor -> 1.0
                progress = step / cfg.warmup_steps
                return cfg.warmup_start_factor + (1.0 - cfg.warmup_start_factor) * progress

            # cosine decay: 1.0 -> min_lr_factor over the remaining steps
            decay_steps = max(1, cfg.total_steps - cfg.warmup_steps)
            progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cfg.min_lr_factor + (1.0 - cfg.min_lr_factor) * cosine

        return fn


@Scheduler.register(config=WarmupCosineConfig())
def warmup_cosine(optimizer: optim.Optimizer, cfg: WarmupCosineConfig) -> WarmupCosine:
    return WarmupCosine(optimizer, cfg)
