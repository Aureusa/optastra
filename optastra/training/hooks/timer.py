import time
from .base import Hook
from ..state import TrainerState


class IterTimerHook(Hook):
    """Writes 'time' (full step) and 'data_time' (batch-fetch portion) into
    storage every iteration. Trainer doesn't need to know timing exists."""

    def before_step(self, state: TrainerState) -> None:
        self._step_start = time.perf_counter()
        # data_time is measured as "time since previous after_step" by the
        # trainer's data-fetch happening between before_step and the actual
        # forward call; simplest correct approach: Trainer records batch-fetch
        # timing itself and pushes it, since only Trainer knows where fetch ends.
        state.storage.put_scalar("data_time", getattr(state, "_last_data_time", 0.0))

    def after_step(self, state: TrainerState) -> None:
        state.storage.put_scalar("time", time.perf_counter() - self._step_start)
