from __future__ import annotations
from typing import Any

from ..core.factory import Factory
from ..core.registry import FamilyRegistry
from ..nn.features import HeadOutput


class DetectionCriterion(Factory["DetectionCriterion"]):
    """Base interface for task-pluggable detection loss/metric implementations."""

    required_fields: tuple[str, ...] = ()
    _registry = FamilyRegistry("criterion")

    def validate_predictions(self, raw_preds: HeadOutput) -> None:
        missing = [field for field in self.required_fields if getattr(raw_preds, field, None) is None]
        if missing:
            raise ValueError(f"Detection criterion requires fields {missing}, got {raw_preds}.")

    def compute_losses(self, raw_preds: HeadOutput, targets: Any) -> dict[str, Any]:
        raise NotImplementedError

    def compute_metrics(self, raw_preds: HeadOutput, targets: Any) -> dict[str, float]:
        return {}
