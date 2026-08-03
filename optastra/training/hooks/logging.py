import logging
from .base import Hook
from ..state import TrainerState


class ConsoleLoggerHook(Hook):
    def __init__(self, log_every: int = 20):
        self.log_every = log_every
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True

    def after_step(self, state: TrainerState) -> None:
        if state.iter % self.log_every != 0:
            return
        loss = state.storage.smoothed("loss")
        self.logger.info(f"iter {state.iter}/{state.max_iter}  loss={loss:.4f}")

    def after_eval_step(self, state: TrainerState) -> None:
        max_eval_iter = state.storage.max_eval_iter
        if max_eval_iter == 0:
            self.logger.warning("max_eval_iter is 0, cannot log eval metrics.")
            return
        if state.storage.eval_iter % self.log_every != 0:
            return
        
        eval_metrics = state.storage.latest_fresh(max_age=0)
        eval_metrics = {k: v for k, v in eval_metrics.items() if k.startswith("val_")}

        # Get counter for eval metrics, which is the eval_iter
        eval_iter = state.storage.eval_iter
        if eval_metrics:
            eval_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(eval_metrics.items()))
            self.logger.info(f"[eval @ iter {state.iter}] iter {eval_iter}/{max_eval_iter}  {eval_str}")
