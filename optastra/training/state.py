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
    iter: int = 0
    epoch: int = 0
    max_iter: int = 0
    eval_iter: int = 0
    max_eval_iter: int = 0
    last_output: TaskStepOutput | None = None
    last_data_time: float = 0.0
    should_stop: bool = False
    