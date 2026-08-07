from __future__ import annotations
from ..architectures.base import Architecture
from ..backbones import Backbone
from ..necks import Neck
from ..heads import Head
from ..tasks import Task
from ..optim import Optimizer, Scheduler
from .experiment import ExperimentConfig
from .component_ref import ComponentRef, coerce_component_refs


def build_experiment_from_config(cfg: ExperimentConfig) -> dict:
    model = Architecture.create(cfg.architecture.name, **cfg.architecture.overrides)
    task = Task.create(cfg.task.name, **cfg.task.overrides)
    optimizer = Optimizer.create(cfg.optimizer.name, model, **cfg.optimizer.overrides)
    scheduler = Scheduler.create(cfg.scheduler.name, optimizer, **cfg.scheduler.overrides) if cfg.scheduler else None
    return {"model": model, "task": task, "optimizer": optimizer, "scheduler": scheduler}


@coerce_component_refs
def build_sequential_model(
        backbone: ComponentRef,
        necks: list[ComponentRef],
        head: ComponentRef
    ) -> Architecture:
    """
    Build a sequential model architecture from the given backbone, necks, and head.
    The model is structured as follows:
    Backbone-> Neck(s) -> Head

    :param backbone: ComponentRef for the backbone.
    :param necks: List of ComponentRefs for the necks.
    :param head: ComponentRef for the head.
    """
    from torch import nn
    bb = Backbone.create(backbone.name, **backbone.overrides)
    out_spec = bb.out_spec
    neck_modules = []
    for neck in necks:
        neck_module = Neck.create(neck.name, **neck.overrides, in_spec=out_spec)
        neck_modules.append(neck_module)
        out_spec = neck_module.out_spec
    head_module = Head.create(head.name, **head.overrides, in_spec=out_spec)
    return nn.Sequential(
        *([bb] + neck_modules + [head_module])
    )
    