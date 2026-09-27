from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn

from ..tasks.base import Task, TaskStepOutput
from .storage import EventStorage


@dataclass
class TrainerState:
    model: nn.Module
    task: Task
    optimizer: Any
    storage: EventStorage
    device: torch.device
    current_batch: dict[str, Any] = field(default_factory=dict)
    hooks: list[Any] = field(default_factory=list)
    iter: int = 0           # optimizer steps (one iter may consume several micro-batches)
    start_iter: int = 0     # first iter train() runs; advanced by ResumeHook via Checkpointer.load
    micro_step: int = 0     # index of state.current_batch within the current iter (grad accumulation)
    epoch: int = 0
    max_iter: int = 0
    eval_iter: int = 0
    max_eval_iter: int = 0
    # Prefixed dataset-level results of the most recent Trainer.evaluate()
    # call (e.g. {"val_accuracy": 0.9, "val_total_loss": 0.3}); set before
    # the after_eval hooks run. Also written to storage on the "iter" axis.
    eval_results: dict[str, float] = field(default_factory=dict)
    last_output: TaskStepOutput | None = None
    last_data_time: float = 0.0
    should_stop: bool = False
