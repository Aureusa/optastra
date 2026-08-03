from dataclasses import dataclass
import torch.optim as optim
from ._registry import register_optimizer


@dataclass
class SGDConfig:
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True

@register_optimizer(config=SGDConfig())
def sgd(param_groups, cfg: SGDConfig) -> optim.SGD:
    return optim.SGD(param_groups, lr=cfg.lr, momentum=cfg.momentum,
                      weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)

