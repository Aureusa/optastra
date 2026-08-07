from __future__ import annotations
from abc import ABC
from typing import Any
import torch

from .base import Task
from ..nn.features import HeadOutput


class CriterionBasedTask(Task, ABC):
    """Task family whose loss/metric/decode logic is fully owned by a
    Criterion + Postprocessor component, not by the Task itself. Detection,
    segmentation, and future criterion-driven tasks subclass this and only
    implement the batch-shape methods (validate_batch, split_inputs_targets,
    preprocess_targets, forward_model) -- which genuinely differ per task
    family and don't belong behind a config flag."""

    criterion: Any    # set by subclass __init__ via resolve_component(cfg, "criterion", ...)
    postprocessor: Any | None = None

    def validate_predictions(self, raw_preds: Any) -> None:
        if not isinstance(raw_preds, HeadOutput):
            raise TypeError(f"Model output must be a HeadOutput, got {type(raw_preds)}.")
        self.criterion.validate_predictions(raw_preds)

    def compute_losses(self, raw_preds: HeadOutput, targets: Any) -> dict[str, torch.Tensor]:
        return self.criterion.compute_losses(raw_preds, targets)

    def reduce_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(losses.values())

    def compute_metrics(self, raw_preds: HeadOutput, targets: Any) -> dict[str, float]:
        return self.criterion.compute_metrics(raw_preds, targets)

    def decode_predictions(self, raw_preds: HeadOutput) -> Any:
        if self.postprocessor is None:
            return raw_preds
        return self.postprocessor.process(raw_preds) # TODO: This expects num_classes to be passed as well...
    