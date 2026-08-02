from __future__ import annotations
from typing import Callable
import logging

from .base import Hook
from ..state import TrainerState


class EvalHook(Hook):
    """
    Periodically runs a no-arg eval function and pushes results into
    EventStorage under a prefix. Doesn't know what's being evaluated --
    just calls eval_fn() and logs whatever dict it returns.
    """
    def __init__(
        self,
        eval_period: int,
        eval_fn: Callable[[], dict[str, float]],
        prefix: str = "val",
        eval_after_train: bool = True,
    ):
        self.eval_period = eval_period
        self.eval_fn = eval_fn
        self.prefix = prefix
        self.eval_after_train = eval_after_train
        self.logger = logging.getLogger("optastra.eval")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
            self.logger.addHandler(handler)

        self.logger.propagate = False

    def _do_eval(self, state: TrainerState) -> None:
        metrics = self.eval_fn()
        state.storage.put_scalars(**{f"{self.prefix}_{k}": v for k, v in metrics.items()})
        if metrics:
            metrics_str = "  ".join(f"{self.prefix}_{k}={v:.4f}" for k, v in metrics.items())
            self.logger.info(f"eval at iter {state.iter}/{state.max_iter}  {metrics_str}")
        else:
            self.logger.info(f"eval at iter {state.iter}/{state.max_iter} produced no metrics")

    def after_step(self, state: TrainerState) -> None:
        # Fire at iter 200, 400, ... for eval_period=200 and skip iter 0.
        if self.eval_period > 0 and state.iter > 0 and state.iter % self.eval_period == 0:
            self._do_eval(state)

    def after_train(self, state: TrainerState) -> None:
        if self.eval_after_train:
            self._do_eval(state)
            