from dataclasses import dataclass
import torch.optim as optim

from ._registry import register_optimizer
from .base import Optimizer


@dataclass
class AdamWConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8

class AdamW(Optimizer, optim.AdamW):
    """Wrapper for torch.optim.AdamW with default parameters."""
    def __init__(self, param_groups, cfg: AdamWConfig):
        super().__init__(
            param_groups,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay
        )
        self.cfg = cfg

@register_optimizer(config=AdamWConfig())
def adamw(param_groups, cfg: AdamWConfig) -> optim.AdamW:
    return AdamW(param_groups, cfg=cfg)
