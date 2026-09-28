from dataclasses import dataclass
import torch

from .base import Optimizer


__all__ = ["LARS", "LARSConfig", "lars"]


@dataclass
class LARSConfig:
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1.5e-6
    eta: float = 1e-3               # trust coefficient
    # Param groups with weight_decay == 0 (biases, norm layers -- see
    # build_param_groups) skip the trust ratio, as in BYOL / SimCLR.
    exclude_zero_decay: bool = True


class LARS(torch.optim.Optimizer):
    """LARS (You et al. 2017, "Large Batch Training of Convolutional Networks"),
    the optimizer of the large-batch self-supervised recipes (SimCLR, BYOL).

    SGD with momentum where each parameter tensor's update is rescaled by a
    trust ratio ``eta * ||w|| / ||g + wd * w||``, so every layer moves by a
    similar *relative* amount whatever the scale of its gradient:

        u   = g + wd * w
        u   = u * eta * ||w|| / ||u||      (groups with lars_adapt only)
        buf = momentum * buf + u
        w   = w - lr * buf

    Biases and normalization parameters get neither weight decay nor the
    trust ratio: optastra's ``build_param_groups`` already puts them in
    groups with ``weight_decay=0``, and with ``exclude_zero_decay`` those
    groups are not adapted (each group's ``lars_adapt`` flag).
    """

    def __init__(self, param_groups, cfg: LARSConfig = LARSConfig()):
        defaults = dict(lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, eta=cfg.eta)
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            group.setdefault("lars_adapt", not (cfg.exclude_zero_decay and group["weight_decay"] == 0))
        self.cfg = cfg

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                update = p.grad
                if group["weight_decay"] != 0:
                    update = update.add(p, alpha=group["weight_decay"])
                if group["lars_adapt"]:
                    w_norm = torch.linalg.vector_norm(p)
                    u_norm = torch.linalg.vector_norm(update)
                    # No rescaling while either norm is 0 (e.g. zero-initialized weights).
                    trust = torch.where(
                        (w_norm > 0) & (u_norm > 0), group["eta"] * w_norm / u_norm, torch.ones_like(w_norm)
                    )
                    update = update * trust
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = update.clone()
                else:
                    state["momentum_buffer"].mul_(group["momentum"]).add_(update)
                p.add_(state["momentum_buffer"], alpha=-group["lr"])
        return loss


@Optimizer.register(config=LARSConfig())
def lars(param_groups, cfg: LARSConfig) -> LARS:
    return LARS(param_groups, cfg)
