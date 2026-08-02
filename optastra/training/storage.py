from __future__ import annotations
from collections import defaultdict, deque
from typing import Any


class EventStorage:
    """
    Generic scalar/metric bus. Hooks read from here keeping hooks decoupled
    from any particular Task's output shape. Anything (loss, lr, grad norm, custom metric)
    is just a named scalar with a history.
    This is a simplified version of Detectron2's EventStorage.

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
        self._latest: dict[str, float] = {}
        self.iter: int = start_iter

    def put_scalar(self, name: str, value: float) -> None:
        value = float(value)
        self._history[name].append((self.iter, value))
        self._latest[name] = value

    def put_scalars(self, **kwargs: float) -> None:
        for name, value in kwargs.items():
            self.put_scalar(name, value)

    def latest(self) -> dict[str, float]:
        return dict(self._latest)

    def smoothed(self, name: str) -> float:
        """Windowed average -- smooths noisy per-step values for logging."""
        vals = [v for _, v in self._history[name]]
        return sum(vals) / len(vals) if vals else float("nan")

    def history(self, name: str) -> list[tuple[int, float]]:
        return list(self._history[name])
    