import logging
from .base import Hook
from ..state import TrainerState


class ConsoleLoggerHook(Hook):
    """Minimal console logger: smoothed train loss every `log_every` iters and
    the per-batch eval scalars every `log_every` eval batches."""

    def __init__(self, log_every: int = 20):
        self.log_every = log_every
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True

    def after_step(self, state: TrainerState) -> None:
        if state.iter % self.log_every != 0:
            return
        loss = state.storage.smoothed("total_loss")
        self.logger.info(f"iter {state.iter}/{state.max_iter}  total_loss={loss:.4f}")

    def after_eval_step(self, state: TrainerState) -> None:
        storage = state.storage
        if storage.eval_iter % self.log_every != 0:
            return

        eval_metrics = storage.latest_fresh(max_age=0, axis="eval_iter")
        eval_metrics = {k: v for k, v in eval_metrics.items() if k not in ("eval_time", "eval_data_time")}
        total = storage.max_eval_iter or "?"
        if eval_metrics:
            eval_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(eval_metrics.items()))
            self.logger.info(f"[eval @ iter {state.iter}] batch {storage.eval_iter + 1}/{total}  {eval_str}")
