# optim/scheduler_base.py
from __future__ import annotations
from dataclasses import replace, fields
from typing import Any
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ._registry import (
    get_scheduler_entrypoint, get_scheduler_default_config,
    list_schedulers, check_scheduler_registered,
)


__all__ = ["Scheduler"]


class Scheduler:
    """Factory only, same shape as Optimizer.create -- constructs a real
    torch.optim.lr_scheduler.LRScheduler, doesn't wrap or replace it."""

    @classmethod
    def create(cls, name: str, optimizer: optim.Optimizer, **overrides) -> lr_scheduler.LRScheduler:
        if not check_scheduler_registered(name):
            raise ValueError(f"Scheduler '{name}' is not registered.")

        entrypoint = get_scheduler_entrypoint(name)
        default_cfg = get_scheduler_default_config(name)
        cfg = replace(default_cfg, **overrides)
        return entrypoint(optimizer, cfg)

    @classmethod
    def describe(cls, name: str) -> None:
        cfg = get_scheduler_default_config(name)
        print(f"{name}:")
        for f in fields(cfg):
            print(f"  {f.name}: {f.type} = {getattr(cfg, f.name)!r}")

    @classmethod
    def config(cls, name: str) -> Any:
        return get_scheduler_default_config(name)

    @classmethod
    def list_schedulers(cls, filter: str | None = None) -> list[str]:
        return list_schedulers(filter=filter)
    