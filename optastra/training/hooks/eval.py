from __future__ import annotations
from typing import Callable
import logging

from .base import Hook
from ..state import TrainerState


class EvalHook(Hook):
    """
    Periodically runs a no-arg eval function, typically
    ``lambda: trainer.evaluate(val_loader)``.

    Storage is Trainer.evaluate()'s job, not this hook's: evaluate() writes
    the dataset-level summaries (``val_accuracy``, ``val_total_loss``, ...)
    to storage once, keeps per-batch eval scalars in a separate namespace
    (so they never pollute the training smoothing windows), and runs the
    ``*_eval`` hooks. This hook only decides *when* to evaluate and logs the
    returned dict. A custom ``eval_fn`` that doesn't go through
    Trainer.evaluate() therefore writes nothing to storage.
    """
    def __init__(
        self,
        eval_period: int,
        eval_fn: Callable[[], dict[str, float]],
        prefix: str = "val",           # only used in the log line
        eval_after_train: bool = True,
    ):
        self.eval_period = eval_period
        self.eval_fn = eval_fn
        self.prefix = prefix
        self.eval_after_train = eval_after_train
        self._last_eval_iter: int | None = None
        self.logger = logging.getLogger("optastra.eval")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = True

    def _do_eval(self, state: TrainerState) -> None:
        self._last_eval_iter = state.iter
        metrics = self.eval_fn()
        if metrics:
            metrics_str = "  ".join(f"{self.prefix}_{k}={v:.4f}" for k, v in metrics.items())
            self.logger.info(f"eval at iter {state.iter}/{state.max_iter}  {metrics_str}")
        else:
            self.logger.info(f"eval at iter {state.iter}/{state.max_iter} produced no metrics")

    def after_step(self, state: TrainerState) -> None:
        # Fire at iter 200, 400, ... for eval_period=200 and skip iter 0.
        if self.eval_period > 0 and state.iter > 0 and state.iter % self.eval_period == 0:
            self.logger.info(f"Running evaluation at iter {state.iter}/{state.max_iter}...")
            self._do_eval(state)

    def after_train(self, state: TrainerState) -> None:
        # Skip if the periodic eval already ran on these exact weights.
        if self.eval_after_train and self._last_eval_iter != state.iter:
            self._do_eval(state)
