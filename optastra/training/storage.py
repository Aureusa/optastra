from __future__ import annotations
from collections import defaultdict, deque


AXES = ("iter", "eval_iter")


class EventStorage:
    """
    Generic scalar/metric bus. Hooks read from here keeping hooks decoupled
    from any particular Task's output shape. Anything (loss, lr, grad norm, custom metric)
    is just a named scalar with a history.

    Scalars live on one of two separate axes (namespaces):

    - ``axis="iter"`` (default): tagged with the training step. Train scalars
      (``total_loss``, ``lr``, ``grad_norm``, ...) and the per-eval *summaries*
      (``val_accuracy``, ``val_total_loss``, ...) written once per
      ``Trainer.evaluate()`` call. These persist for the whole run.
    - ``axis="eval_iter"``: tagged with the eval-batch index. Per-batch eval
      scalars (``val_step_*``, ``eval_time``, ...). Cleared by
      ``reset_eval()`` at the start of every evaluation, so they never mix
      into the training smoothing windows or leak from one eval into the next.

    Inspired by Detectron2:
    @misc{wu2019detectron2,
    author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                    Wan-Yen Lo and Ross Girshick},
    title =        {Detectron2},
    howpublished = {https://github.com/facebookresearch/detectron2},
    year =         {2019}
    }
    """

    def __init__(self, start_iter: int = 0, window_size: int = 20):
        self.window_size = window_size
        self._history: dict[str, dict[str, deque[tuple[int, float]]]] = {axis: self._new_history() for axis in AXES}
        self._latest: dict[str, dict[str, tuple[int, float]]] = {axis: {} for axis in AXES}
        self.iter: int = start_iter        # training step counter
        self.eval_iter: int = 0            # eval batch counter, resets each eval() call
        self.max_eval_iter = 0             # number of eval batches (0 if the loader has no len())

    def _new_history(self) -> defaultdict[str, deque[tuple[int, float]]]:
        return defaultdict(lambda: deque(maxlen=self.window_size))

    @staticmethod
    def _check_axis(axis: str) -> None:
        if axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {axis!r}")

    def _now(self, axis: str) -> int:
        return self.iter if axis == "iter" else self.eval_iter

    def put_scalar(self, name: str, value: float, *, axis: str = "iter") -> None:
        """axis='iter' -> tag with training step (default, for train/summary metrics).
        axis='eval_iter' -> tag with eval-batch index (for per-batch eval metrics)."""
        self._check_axis(axis)
        value = float(value)
        tag = self._now(axis)
        self._history[axis][name].append((tag, value))
        self._latest[axis][name] = (tag, value)

    def put_scalars(self, *, axis: str = "iter", **kwargs: float) -> None:
        for name, value in kwargs.items():
            self.put_scalar(name, value, axis=axis)

    def reset_eval(self) -> None:
        """Forget every eval_iter-axis scalar; called at the start of each evaluation."""
        self._history["eval_iter"] = self._new_history()
        self._latest["eval_iter"] = {}
        self.eval_iter = 0

    def keys(self, *, axis: str = "iter") -> list[str]:
        self._check_axis(axis)
        return list(self._latest[axis].keys())

    def has(self, name: str, *, axis: str = "iter") -> bool:
        self._check_axis(axis)
        return name in self._latest[axis]

    def latest(self, *, axis: str = "iter") -> dict[str, float]:
        self._check_axis(axis)
        return {k: v for k, (_, v) in self._latest[axis].items()}

    def latest_fresh(self, max_age: int = 0, *, axis: str = "iter") -> dict[str, float]:
        """Latest values written within the last `max_age` steps of `axis`."""
        self._check_axis(axis)
        now = self._now(axis)
        return {k: v for k, (tag, v) in self._latest[axis].items() if now - tag <= max_age}

    def smoothed(self, name: str, *, axis: str = "iter") -> float:
        """Mean over the last `window_size` values of `name` (nan if never written)."""
        self._check_axis(axis)
        vals = [v for _, v in self._history[axis].get(name, ())]
        return sum(vals) / len(vals) if vals else float("nan")

    def history(self, name: str, *, axis: str = "iter") -> list[tuple[int, float]]:
        self._check_axis(axis)
        return list(self._history[axis].get(name, ()))
