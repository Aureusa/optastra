from __future__ import annotations

from ..core.factory import Factory
from ..data.sample import Sample
from ..nn.features import HeadOutput

from ..core.registry import FamilyRegistry


class Postprocessor(Factory["Postprocessor"]):
    """Factory for postprocessors that convert raw model outputs into structured predictions."""

    _registry = FamilyRegistry("postprocessor")

    def process(self, raw_preds: HeadOutput, num_classes: int) -> list[Sample]:
        raise NotImplementedError("Postprocessor subclasses must implement the process method.")
    