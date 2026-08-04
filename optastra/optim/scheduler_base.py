# optim/scheduler_base.py
from __future__ import annotations
from dataclasses import replace, fields
from typing import Any
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ._registry import _scheduler_registry
from ..core.factory import Factory


__all__ = ["Scheduler"]


class Scheduler(Factory["Scheduler"]):
    """Factory only, same shape as Optimizer.create -- constructs a real
    torch.optim.lr_scheduler.LRScheduler, doesn't wrap or replace it."""

    _registry = _scheduler_registry

    @classmethod
    def create(cls, name: str, optimizer: optim.Optimizer, **overrides) -> lr_scheduler.LRScheduler:
        cls._check_registered(name)

        entrypoint = cls._registry.get_entrypoint(name)
        default_cfg = cls._registry.get_default_config(name)
        cfg = replace(default_cfg, **overrides)
        return entrypoint(optimizer, cfg)
    