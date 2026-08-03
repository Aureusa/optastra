from __future__ import annotations
import copy
from collections import defaultdict, deque
from typing import Any


class EventStorage:
    """
    Generic scalar/metric bus. Hooks read from here keeping hooks decoupled
    from any particular Task's output shape. Anything (loss, lr, grad norm, custom metric)
    is just a named scalar with a history.
    
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
        self._history: dict[str, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._latest: dict[str, tuple[int, float]] = {}
        self.iter: int = start_iter        # training step counter
        self.eval_iter: int = 0            # eval batch counter, resets each eval() call
        self.max_eval_iter = 0  # max eval batch counter, set at start of each eval() call

    def put_scalar(self, name: str, value: float, *, axis: str = "iter") -> None:
        """axis='iter' -> tag with training step (default, for train/summary metrics).
        axis='eval_iter' -> tag with eval-batch index (for per-batch eval metrics)."""
        value = float(value)
        tag = self.iter if axis == "iter" else self.eval_iter
        self._history[name].append((tag, value))
        self._latest[name] = (tag, value)

    def keys(self) -> list[str]:
        return list(self._history.keys())

    def put_scalars(self, *, axis: str = "iter", **kwargs: float) -> None:
        for name, value in kwargs.items():
            self.put_scalar(name, value, axis=axis)

    def latest(self) -> dict[str, float]:
        return {k: v for k, (_, v) in self._latest.items()}

    def latest_fresh(self, max_age: int = 0, *, axis: str = "iter") -> dict[str, float]:
        now = self.iter if axis == "iter" else self.eval_iter
        return {k: v for k, (tag, v) in self._latest.items() if now - tag <= max_age}

    def smoothed(self, name: str) -> float:
        vals = [v for _, v in self._history[name]]
        return sum(vals) / len(vals) if vals else float("nan")

    def history(self, name: str) -> list[tuple[int, float]]:
        return list(self._history[name])

    def snapshot(self) -> dict[str, Any]:
        return {
            "_history": copy.deepcopy(self._history),
            "_latest": copy.deepcopy(self._latest),
            "iter": self.iter,
            "eval_iter": self.eval_iter,
            "max_eval_iter": self.max_eval_iter,
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self._history = snapshot["_history"]
        self._latest = snapshot["_latest"]
        self.iter = snapshot["iter"]
        self.eval_iter = snapshot["eval_iter"]
        self.max_eval_iter = snapshot["max_eval_iter"]
        