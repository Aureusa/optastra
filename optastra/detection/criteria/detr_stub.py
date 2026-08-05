from __future__ import annotations

from dataclasses import dataclass

from .._registry import register_criterion
from ..base_criterion import DetectionCriterion
from ...nn.features import HeadOutput


@dataclass
class DETRCriterionStubConfig:
    num_classes: int = 80


class DETRCriterionStub(DetectionCriterion):
    """Registration stub that defines the transformer-style criterion contract.

    Use this as a plug point for future Hungarian matching + set-based losses.
    """

    required_fields = ("logits", "boxes")

    def validate_predictions(self, raw_preds: HeadOutput) -> None:
        super().validate_predictions(raw_preds)

    def compute_losses(self, raw_preds: HeadOutput, targets):
        raise NotImplementedError(
            "detr_criterion_stub is a contract placeholder. "
            "Register a concrete DETR criterion implementation to train transformer detectors."
        )


criterion_configs = {
    "detr_criterion_stub": DETRCriterionStubConfig(),
}


@register_criterion(config=criterion_configs["detr_criterion_stub"])
def detr_criterion_stub(cfg: DETRCriterionStubConfig) -> DETRCriterionStub:
    return DETRCriterionStub()
