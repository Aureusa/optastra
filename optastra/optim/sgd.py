from dataclasses import dataclass
import torch.optim as optim

from .base import Optimizer


__all__ = ["SGD", "SGDConfig", "sgd"]


@dataclass
class SGDConfig:
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True


class SGD(optim.SGD):
    """torch.optim.SGD built from an SGDConfig. The per-group
    weight_decay set by build_param_groups takes precedence over
    cfg.weight_decay (which only becomes the optimizer-level default)."""

    def __init__(self, param_groups, cfg: SGDConfig):
        super().__init__(param_groups, lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
        self.cfg = cfg


@Optimizer.register(config=SGDConfig())
def sgd(param_groups, cfg: SGDConfig) -> SGD:
    return SGD(param_groups, cfg)
