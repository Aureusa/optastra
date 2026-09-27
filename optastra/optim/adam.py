from dataclasses import dataclass
import torch.optim as optim

from .base import Optimizer


__all__ = ["Adam", "AdamConfig", "adam"]


@dataclass
class AdamConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0   # coupled L2 (added to the gradient); prefer adamw for decoupled decay
    eps: float = 1e-8


class Adam(optim.Adam):
    """torch.optim.Adam built from an AdamConfig. The per-group
    weight_decay set by build_param_groups takes precedence over
    cfg.weight_decay (which only becomes the optimizer-level default)."""

    def __init__(self, param_groups, cfg: AdamConfig):
        super().__init__(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                         weight_decay=cfg.weight_decay)
        self.cfg = cfg


@Optimizer.register(config=AdamConfig())
def adam(param_groups, cfg: AdamConfig) -> Adam:
    return Adam(param_groups, cfg)
