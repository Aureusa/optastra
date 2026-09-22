import logging
from .base import Hook
from .best_metric import BestMetricTracker, resolve_tracker
from ..state import TrainerState


class EarlyStoppingHook(Hook):
    """Flags state.should_stop -- Trainer checks it, doesn't need to know why.
    Asks a BestMetricTracker whether the monitored metric (e.g. 'val_total_loss')
    improved, once per eval call via after_eval. Pass a shared `tracker` to
    agree with BestCheckpointHook on what "best" means.
    """

    def __init__(
            self,
            metric: str | None = None,
            patience: int = 10,
            mode: str | None = None,
            reset: bool = False,
            tracker: BestMetricTracker | None = None,
        ):
        self.tracker = resolve_tracker(tracker, metric, mode)
        self.patience = patience
        self.bad_evals = 0
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True
        self.reset_flag = reset

    def before_train(self, state: TrainerState) -> None:
        self.logger.info(
            f"EarlyStoppingHook initialized: monitoring '{self.tracker.metric}' with patience={self.patience}"
            f" and mode='{self.tracker.mode}'."
        )
        if not self.reset_flag:
            self._condition_met(state)  # check if early stopping condition is already met at the start of training
        else:
            self.tracker.reset()  # NOTE: also resets "best" for any hook sharing this tracker
            self.bad_evals = 0
            self.logger.info(f"EarlyStoppingHook reset because `reset` flag is set to `True`.")

    def after_eval(self, state: TrainerState) -> None:
        improved = self.tracker.update(state)
        if improved is None:
            return  # metric not written this eval call -- tracker already warned
        self.bad_evals = 0 if improved else self.bad_evals + 1
        self._condition_met(state)

    def _condition_met(self, state: TrainerState) -> None:
        if self.bad_evals >= self.patience:
            self.logger.info(f"Early stopping triggered: {self.tracker.metric} did not improve for {self.patience} evals.")
            state.should_stop = True

    def state_dict(self) -> dict:
        return {"tracker": self.tracker.state_dict(), "bad_evals": self.bad_evals}

    def load_state_dict(self, state: dict) -> None:
        # Checkpoints written before the tracker refactor stored "best" at the top level.
        self.tracker.load_state_dict(state.get("tracker", {"best": state["best"]}))
        self.bad_evals = state["bad_evals"]
        self.logger.info(f"EarlyStoppingHook state loaded: best={self.tracker.best}, bad_evals={self.bad_evals}.")
