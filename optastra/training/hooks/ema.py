import torch
from .base import Hook
from ..state import TrainerState

class EMAHook(Hook):
    """
    Updates a teacher module's weights as an EMA of a student module's,
    after every optimizer step. Trainer/Task never know this is happening.
    """

    def __init__(self, student: torch.nn.Module, teacher: torch.nn.Module, momentum: float = 0.996):
        self.student, self.teacher, self.momentum = student, teacher, momentum
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def after_step(self, state: TrainerState) -> None:
        m = self.momentum
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps, alpha=1 - m)