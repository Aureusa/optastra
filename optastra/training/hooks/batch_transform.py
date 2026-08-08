from __future__ import annotations
from .base import Hook
from ..state import TrainerState
from ...transforms.batch import BatchTransform


class BatchTransformHook(Hook):
    """Applies one or more BatchTransforms to the fetched batch before
    Task.run_step sees it. Order matters -- e.g. CutMix then Normalize."""

    def __init__(self, transforms: list[BatchTransform]):
        self.transforms = transforms

    def before_step(self, state: TrainerState) -> None:
        batch = state.current_batch
        for t in self.transforms:
            batch = t(batch)
        state.current_batch = batch
        