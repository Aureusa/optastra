from __future__ import annotations

from ..core.factory import Factory
from ..data.sample import Sample
from ..nn.features import HeadOutput

from ._registry import postprocessor_registry


class Postprocessor(Factory["Postprocessor"]):
    """Factory for postprocessors that convert raw model outputs into structured predictions."""

    _registry = postprocessor_registry

    def process(self, raw_preds: HeadOutput, num_classes: int) -> list[Sample]:
        raise NotImplementedError("Postprocessor subclasses must implement the process method.")
    