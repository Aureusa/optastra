from dataclasses import dataclass
import torch.optim as optim

from .base import Optimizer


__all__ = ["AdamW", "AdamWConfig", "adamw"]


@dataclass
class AdamWConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8


class AdamW(optim.AdamW):
    """torch.optim.AdamW built from an AdamWConfig. The per-group
    weight_decay set by build_param_groups takes precedence over
    cfg.weight_decay (which only becomes the optimizer-level default)."""

    def __init__(self, param_groups, cfg: AdamWConfig):
        super().__init__(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                         weight_decay=cfg.weight_decay)
        self.cfg = cfg


@Optimizer.register(config=AdamWConfig())
def adamw(param_groups, cfg: AdamWConfig) -> AdamW:
    return AdamW(param_groups, cfg)
