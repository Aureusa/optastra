from __future__ import annotations
import torch

from ..core.factory import Factory
from ._registry import matcher_registry


class Matcher(Factory["Matcher"]):
    """Match proposals/anchors to GT using IoU thresholds."""

    _registry = matcher_registry

    def match(
        self,
        proposals: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        *,
        background_label: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Matcher subclasses must implement the match method.")
    