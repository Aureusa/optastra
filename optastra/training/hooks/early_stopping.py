from .base import Hook
from ..state import TrainerState


class EarlyStoppingHook(Hook):
    """Flags state.should_stop -- Trainer checks it, doesn't need to know why."""
    def __init__(self, metric: str = "loss", patience: int = 5, mode: str = "min"):
        self.metric, self.patience, self.mode = metric, patience, mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.bad_epochs = 0

    def after_epoch(self, state: TrainerState) -> None:
        current = state.storage.smoothed(self.metric)
        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best, self.bad_epochs = current, 0
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            state.should_stop = True