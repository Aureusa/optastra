from __future__ import annotations
from abc import ABC

from ..state import TrainerState


class Hook(ABC):
    """Observer of the training loop. A hook only ever reads TrainerState /
    EventStorage -- it never touches the model, task, or TaskStepOutput
    directly, so new tasks or new TaskStepOutput fields never require hooks
    to change. Override only the lifecycle methods you need."""

    # Lower runs first within every lifecycle event. Trainer sorts hooks
    # stably by this, so equal-priority hooks keep registration order.
    # 0 = must see state before anyone else (ResumeHook); 100 = must see
    # everyone else's updates for this step (checkpointing).
    priority: int = 50

    def before_train(self, state: TrainerState) -> None: ...
    def after_train(self, state: TrainerState) -> None: ...
    def before_epoch(self, state: TrainerState) -> None: ...
    def after_epoch(self, state: TrainerState) -> None: ...
    def before_step(self, state: TrainerState) -> None: ...
    def after_step(self, state: TrainerState) -> None: ...
    def before_eval(self, state: TrainerState) -> None: ...
    def after_eval(self, state: TrainerState) -> None: ...
    def before_eval_step(self, state: TrainerState) -> None: ...
    def after_eval_step(self, state: TrainerState) -> None: ...

    # optional, for resumable hooks (e.g. checkpoint counters, EMA state)
    def state_dict(self) -> dict: return {}
    def load_state_dict(self, state: dict) -> None: return

    def info(self) -> str:
        return f"{self.__class__.__name__} (module: {self.__class__.__module__})"
    