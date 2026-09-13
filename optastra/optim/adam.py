from dataclasses import dataclass
import torch.optim as optim

from .base import Optimizer


@dataclass
class AdamConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8


class Adam(optim.Adam):
    def __init__(self, param_groups, cfg: AdamConfig):
        super().__init__(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                         weight_decay=cfg.weight_decay)

@Optimizer.register(config=AdamConfig())
def adam(param_groups, cfg: AdamConfig) -> Adam:
    return Adam(param_groups, cfg)
