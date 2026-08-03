from __future__ import annotations
import torch.optim.lr_scheduler as lr_scheduler

from .base import Hook
from ..state import TrainerState


class SchedulerHook(Hook):
    """Steps an LR scheduler after every training iteration. Trainer/Task
    never know a scheduler exists -- same boundary as EMAHook."""

    def __init__(self, scheduler: lr_scheduler.LRScheduler, log_lr: bool = True):
        self.scheduler = scheduler
        self.log_lr = log_lr

    def after_step(self, state: TrainerState) -> None:
        self.scheduler.step()
        if self.log_lr:
            # log the first group's LR -- representative even with multipliers,
            # since every group moves on the same relative curve
            lr = self.scheduler.get_last_lr()[0]
            state.storage.put_scalar("lr", lr)

    def state_dict(self) -> dict:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.scheduler.load_state_dict(state["scheduler"])
