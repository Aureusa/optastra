from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import torch.nn as nn
import torch

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
    last_output: TaskStepOutput | None = None
    should_stop: bool = False   # hooks (e.g. EarlyStopping) flip this
    