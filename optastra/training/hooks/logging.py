import logging
from .base import Hook
from ..state import TrainerState


class ConsoleLoggerHook(Hook):
    def __init__(self, log_every: int = 20):
        self.log_every = log_every
        self.logger = logging.getLogger("optastra.train")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
            self.logger.addHandler(handler)

        # Avoid duplicate logs if the root logger is also configured.
        self.logger.propagate = False

    def after_step(self, state: TrainerState) -> None:
        if state.iter % self.log_every != 0:
            return
        loss = state.storage.smoothed("loss")
        latest = state.storage.latest()
        val_metrics = {k: v for k, v in latest.items() if k.startswith("val_")}
        if val_metrics:
            val_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
            self.logger.info(f"iter {state.iter}/{state.max_iter}  loss={loss:.4f}  {val_str}")
            return
        self.logger.info(f"iter {state.iter}/{state.max_iter}  loss={loss:.4f}")