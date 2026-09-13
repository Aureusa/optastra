from __future__ import annotations
import torch

from ..core.factory import Factory
from ..core.registry import FamilyRegistry


class Matcher(Factory["Matcher"]):
    """Match proposals/anchors to GT using IoU thresholds."""

    _registry = FamilyRegistry("matcher")

    def match(
        self,
        proposals: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        *,
        background_label: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Matcher subclasses must implement the match method.")
    