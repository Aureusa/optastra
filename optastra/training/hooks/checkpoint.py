import logging

from .base import Hook
from .best_metric import BestMetricTracker, resolve_tracker
from ..checkpointer import Checkpointer
from ..state import TrainerState


class CheckpointHook(Hook):
    """
    Saves a resumable checkpoint every `save_every` iterations.
    File naming and format are owned by the Checkpointer.
    """
    priority = 100  # after SchedulerHook/EvalHook/etc., so their state for this iter is saved

    def __init__(self, checkpointer: Checkpointer | str, save_every: int = 500):
        self.checkpointer = Checkpointer.resolve(checkpointer)
        self.save_every = save_every

    def after_step(self, state: TrainerState) -> None:
        if state.iter == 0 or state.iter % self.save_every != 0:
            return
        self.checkpointer.save(state, self.checkpointer.periodic_name(state.iter))


class BestCheckpointHook(Hook):
    """
    Saves a checkpoint whenever the tracked metric improves on an eval,
    overwriting the previous best. Add one per metric you want a "best"
    model for; pass the same `tracker` as EarlyStoppingHook to have both
    agree on what "best" means.
    """
    priority = 100  # after EarlyStoppingHook etc. have updated for this eval

    def __init__(
            self,
            checkpointer: Checkpointer | str,
            metric: str | None = None,
            mode: str | None = None,
            tracker: BestMetricTracker | None = None,
            filename: str | None = None,
        ):
        self.checkpointer = Checkpointer.resolve(checkpointer)
        self.tracker = resolve_tracker(tracker, metric, mode)
        self.filename = filename or self.checkpointer.best_name(self.tracker.metric)
        self.logger = logging.getLogger("optastra.train")

    def after_eval(self, state: TrainerState) -> None:
        if not self.tracker.update(state):
            return
        path = self.checkpointer.save(
            state, self.filename,
            extra={"metric": self.tracker.metric, "value": self.tracker.best},
        )
        self.logger.info(
            f"New best {self.tracker.metric}={self.tracker.best:.4f} at iter {state.iter}, saved to {path}"
        )

    def state_dict(self) -> dict:
        return {"tracker": self.tracker.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if "tracker" in state:
            self.tracker.load_state_dict(state["tracker"])
