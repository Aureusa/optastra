import logging
from .base import Hook
from ..state import TrainerState


class EarlyStoppingHook(Hook):
    """Flags state.should_stop -- Trainer checks it, doesn't need to know why.
    Monitors a metric written by evaluate() (e.g. 'val_total_loss'), checked
    once per eval call via after_eval.
    """

    def __init__(self, metric: str = "val_total_loss", patience: int = 10, mode: str = "min", before_train: bool = True):
        self.metric, self.patience, self.mode = metric, patience, mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.bad_evals = 0
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True
        self.before_train_flag = before_train

    def before_train(self, state: TrainerState) -> None:
        self.logger.info(f"EarlyStoppingHook initialized: monitoring '{self.metric}' with patience={self.patience} and mode='{self.mode}'.")
        if self.before_train_flag:
            self._condition_met(state)  # check if early stopping condition is already met at the start of training
        else:
            self.best = float("inf") if self.mode == "min" else float("-inf")
            self.bad_evals = 0
            self.logger.info(f"EarlyStoppingHook reset because `before_train` flag is set to `False`.")

    def after_eval(self, state: TrainerState) -> None:
        latest = state.storage.latest()
        if self.metric not in latest:
            self.logger.warning(
                f"EarlyStoppingHook: metric '{self.metric}' not found in latest evaluation."
                f" Early stopping check skipped for this evaluation."
                f" If this is unexpected, ensure that the evaluation function writes the metric '{self.metric}' to the storage."
                f" If `{self.metric}` is not the desired metric, please specify the correct metric name when initializing EarlyStoppingHook."
            )
            return  # or log a warning -- metric not written this eval call
        current = latest[self.metric]

        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best, self.bad_evals = current, 0
        else:
            self.bad_evals += 1
        self._condition_met(state)

    def _condition_met(self, state: TrainerState) -> bool:
        if self.bad_evals >= self.patience:
            self.logger.info(f"Early stopping triggered: {self.metric} did not improve for {self.patience} evals.")
            state.should_stop = True

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_evals": self.bad_evals}

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.bad_evals = state["bad_evals"]
        self.logger.info(f"EarlyStoppingHook state loaded: best={self.best}, bad_evals={self.bad_evals}.")
