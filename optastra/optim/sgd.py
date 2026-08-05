from dataclasses import dataclass
import torch.optim as optim
from ._registry import register_optimizer


@dataclass
class SGDConfig:
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True


class SGD(optim.SGD):
    def __init__(self, param_groups, cfg: SGDConfig):
        super().__init__(param_groups, lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)

@register_optimizer(config=SGDConfig())
def sgd(param_groups, cfg: SGDConfig) -> SGD:
    return SGD(param_groups, cfg)

