from __future__ import annotations
from typing import Any, Mapping
from ..tasks.base import Task, Stage


class Algorithm(Task):
    """
    A Task specialization for self-supervised pretraining objectives.

    Structurally identical to Task (same run_step / TaskStepOutput / Trainer
    integration) -- the difference is entirely in what the batch contains
    and what the loss compares: multiple augmented views of the same
    image, no external labels.
    """
    min_views: int = 2   # default minimum number of views required for a batch

    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if "views" not in batch:
            raise ValueError("Algorithm batches must contain a 'views' key: list[Tensor].")
        if len(batch["views"]) < self.min_views:
            raise ValueError(f"{type(self).__name__} requires >= {self.min_views} views.")

    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train"):
        # Reuse the same views as pseudo-targets so Task.run_step executes
        # the normal train/val loss path without requiring external labels.
        return batch["views"], batch["views"]

    def preprocess_targets(self, raw_targets):
        """Pass-through: algorithms treat views as self-supervised targets."""
        return raw_targets

    def decode_predictions(self, raw_preds: Any) -> Any:
        """No-op for algorithms, since there are no external targets."""
        return raw_preds

    def compute_metrics(self, raw_preds, targets) -> dict[str, float]:
        """No-op for algorithms, since there are no external targets."""
        return {}
