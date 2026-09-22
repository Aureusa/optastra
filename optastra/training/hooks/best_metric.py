from __future__ import annotations
import logging

from ..state import TrainerState


class BestMetricTracker:
    """Decides whether a monitored metric improved on the current eval.

    Not a hook: hooks that care about "best" (BestCheckpointHook,
    EarlyStoppingHook) hold a reference and ask it via update(state) from
    their own after_eval. The verdict is computed once per iteration and
    cached, so:
      - hook order doesn't matter (whoever asks first computes it),
      - one tracker shared by several hooks is a single source of truth
        that can't be double-counted or drift.
    Same iteration => same weights => same verdict.

    Share one instance explicitly when hooks should agree on "best":
        tracker = BestMetricTracker("val_total_loss", "min")
        BestCheckpointHook(checkpointer, tracker=tracker)
        EarlyStoppingHook(tracker=tracker, patience=10)
    """

    def __init__(self, metric: str = "val_total_loss", mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.metric = metric
        self.mode = mode
        self.logger = logging.getLogger("optastra.train")
        self.reset()

    def reset(self) -> None:
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self._last_iter: int | None = None
        self._last_verdict: bool | None = None

    def update(self, state: TrainerState) -> bool | None:
        """True if the metric improved at this iteration, False if not,
        None if the metric wasn't written this iteration (nothing to judge)."""
        if self._last_iter == state.iter:
            return self._last_verdict

        # Fresh only: a failed/partial eval must not be judged on the previous eval's value.
        current = state.storage.latest_fresh(max_age=0).get(self.metric)
        if current is None:
            self.logger.warning(
                f"BestMetricTracker: metric '{self.metric}' was not written at iter {state.iter}."
                f" Best metric check skipped for this evaluation."
                f" If this is unexpected, ensure that the evaluation function writes '{self.metric}' to the storage,"
                f" or pass the correct metric name."
            )
            return None

        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best = current
        self._last_iter, self._last_verdict = state.iter, improved
        return improved

    def state_dict(self) -> dict:
        return {"best": self.best}

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]


def resolve_tracker(
        tracker: BestMetricTracker | None,
        metric: str | None,
        mode: str | None,
        default_metric: str = "val_total_loss",
    ) -> BestMetricTracker:
    """Hooks accept either a shared `tracker` or `metric`/`mode` to build a
    private one -- never both, so a passed-in tracker's settings can't be
    silently contradicted."""
    if tracker is not None:
        if metric is not None or mode is not None:
            raise TypeError("Pass either `tracker` or `metric`/`mode`, not both.")
        return tracker
    return BestMetricTracker(metric or default_metric, mode or "min")
