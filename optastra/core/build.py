from __future__ import annotations
from ..architectures.base import Architecture
from ..tasks.base import Task
from ..optim.base import Optimizer
from ..optim.scheduler_base import Scheduler
from .experiment import ExperimentConfig


def build_from_config(cfg: ExperimentConfig) -> dict:
    model = Architecture.create(cfg.architecture.name, **cfg.architecture.overrides)
    task = Task.create(cfg.task.name, **cfg.task.overrides)
    optimizer = Optimizer.create(cfg.optimizer.name, model, **cfg.optimizer.overrides)
    scheduler = Scheduler.create(cfg.scheduler.name, optimizer, **cfg.scheduler.overrides) if cfg.scheduler else None
    return {"model": model, "task": task, "optimizer": optimizer, "scheduler": scheduler}
