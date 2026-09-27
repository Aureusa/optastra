from __future__ import annotations
import torch.nn as nn
import torch.optim as optim

from .param_groups import build_param_groups, ParamGroupConfig
from ..core.factory import Factory
from ..core.registry import FamilyRegistry


__all__ = ["Optimizer"]


class Optimizer(Factory["Optimizer"]):
    """Factory only -- doesn't wrap or replace torch.optim.Optimizer at runtime,
    it just constructs one correctly, including param groups.

    Registered optimizer configs are expected to have `lr` and
    `weight_decay` fields: both are threaded into every param group
    (`build_param_groups`), so per-group LR multipliers and the no-decay
    bucket (norm/bias/pos_embed/...) are applied on top of them."""

    _registry = FamilyRegistry("optimizer")

    @classmethod
    def create(
        cls,
        name: str,
        model: nn.Module,
        *,
        param_groups: ParamGroupConfig = ParamGroupConfig(),
        **overrides,
    ) -> optim.Optimizer:
        cls._check_registered(name)

        entrypoint = cls._registry.get_entrypoint(name)
        cfg = cls._build_cfg(name, overrides)

        groups = build_param_groups(
            model, param_groups, base_lr=cfg.lr, base_weight_decay=getattr(cfg, "weight_decay", 0.0),
        )
        return entrypoint(groups, cfg)
