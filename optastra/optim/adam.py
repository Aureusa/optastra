from dataclasses import dataclass
import torch.optim as optim
from ._registry import register_optimizer


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

@register_optimizer(config=AdamConfig())
def adam(param_groups, cfg: AdamConfig) -> Adam:
    return Adam(param_groups, cfg)
