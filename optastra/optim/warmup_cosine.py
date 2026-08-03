import math
from dataclasses import dataclass
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ._registry import register_scheduler


@dataclass
class WarmupCosineConfig:
    total_steps: int            # required in practice; pass explicitly via overrides
    warmup_steps: int = 500
    warmup_start_factor: float = 0.01   # warmup begins at this fraction of base_lr
    min_lr_factor: float = 0.0          # cosine floor, as a fraction of base_lr


def _warmup_cosine_lambda(cfg: WarmupCosineConfig):
    def fn(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            # linear warmup: warmup_start_factor -> 1.0
            progress = step / max(1, cfg.warmup_steps)
            return cfg.warmup_start_factor + (1.0 - cfg.warmup_start_factor) * progress

        # cosine decay: 1.0 -> min_lr_factor over the remaining steps
        decay_steps = max(1, cfg.total_steps - cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_factor + (1.0 - cfg.min_lr_factor) * cosine

    return fn


@register_scheduler(config=WarmupCosineConfig(total_steps=100_000))
def warmup_cosine(optimizer: optim.Optimizer, cfg: WarmupCosineConfig) -> lr_scheduler.LambdaLR:
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_cosine_lambda(cfg))