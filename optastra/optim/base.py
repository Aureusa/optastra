# optim/base.py
from __future__ import annotations
from dataclasses import replace, fields
from typing import Any
import torch.nn as nn
import torch.optim as optim

from ._registry import get_optimizer_entrypoint, get_optimizer_default_config, list_optimizers, check_optimizer_registered
from .param_groups import build_param_groups, ParamGroupConfig


__all__ = ["Optimizer"]


class Optimizer:
    """Factory only -- doesn't wrap or replace torch.optim.Optimizer at runtime,
    it just constructs one correctly, including param groups."""

    @classmethod
    def create(
        cls,
        name: str,
        model: nn.Module,
        *,
        param_groups: ParamGroupConfig = ParamGroupConfig(),
        **overrides,
    ) -> optim.Optimizer:
        if not check_optimizer_registered(name):
            raise ValueError(f"Optimizer '{name}' is not registered.")

        entrypoint = get_optimizer_entrypoint(name)
        default_cfg = get_optimizer_default_config(name)
        cfg = replace(default_cfg, **overrides)

        groups = build_param_groups(model, param_groups, base_lr=cfg.lr)
        return entrypoint(groups, cfg)

    @classmethod
    def describe(cls, name: str) -> None:
        cfg = get_optimizer_default_config(name)
        print(f"{name}:")
        for f in fields(cfg):
            print(f"  {f.name}: {f.type} = {getattr(cfg, f.name)!r}")

    @classmethod
    def config(cls, name: str) -> Any:
        return get_optimizer_default_config(name)

    @classmethod
    def list_optimizers(cls, filter: str | None = None) -> list[str]:
        return list_optimizers(filter=filter)