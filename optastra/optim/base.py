from __future__ import annotations
from dataclasses import replace
import torch.nn as nn
import torch.optim as optim

from .param_groups import build_param_groups, ParamGroupConfig
from ..core.factory import Factory
from ..core.registry import FamilyRegistry


__all__ = ["Optimizer"]


class Optimizer(Factory["Optimizer"]):
    """Factory only -- doesn't wrap or replace torch.optim.Optimizer at runtime,
    it just constructs one correctly, including param groups."""

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
        default_cfg = cls._registry.get_default_config(name)
        cfg = replace(default_cfg, **overrides)

        groups = build_param_groups(model, param_groups, base_lr=cfg.lr)
        return entrypoint(groups, cfg)
    