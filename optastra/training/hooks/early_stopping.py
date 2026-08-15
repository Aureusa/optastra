from .base import Hook
from ..state import TrainerState


class EarlyStoppingHook(Hook):
    """Flags state.should_stop -- Trainer checks it, doesn't need to know why.
    Monitors a metric written by evaluate() (e.g. 'val_total_loss'), checked
    once per eval call via after_eval.
    """

    def __init__(self, metric: str = "val_total_loss", patience: int = 10, mode: str = "min"):
        self.metric, self.patience, self.mode = metric, patience, mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.bad_evals = 0

    def after_eval(self, state: TrainerState) -> None:
        latest = state.storage.latest()
        if self.metric not in latest:
            return  # or log a warning -- metric not written this eval call
        current = latest[self.metric]

        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best, self.bad_evals = current, 0
        else:
            self.bad_evals += 1
        if self.bad_evals >= self.patience:
            state.should_stop = True

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_evals": self.bad_evals}

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.bad_evals = state["bad_evals"]
