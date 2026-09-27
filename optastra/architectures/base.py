from __future__ import annotations
from abc import ABC
from typing import Any, Sequence
import torch
import torch.nn as nn

from ..core.factory import Factory
from ..core.registry import FamilyRegistry


__all__ = ["Architecture", "resolve_image_sizes", "add_gt_boxes_to_rois"]


class Architecture(nn.Module, Factory["Architecture"], ABC):

    _registry = FamilyRegistry("architecture")

    def forward(self, x: Any) -> Any:
        raise NotImplementedError("Subclasses must implement the forward method.")


def resolve_image_sizes(
    images: torch.Tensor,
    image_sizes: Sequence[Sequence[int]] | torch.Tensor | None = None,
) -> list[tuple[int, int]]:
    """Per-image (h, w) of the real image content inside a (possibly padded) batch.

    ``image_sizes`` comes from the ragged collate (one entry per image). Without
    it every image is assumed to fill the whole batch tensor.
    """
    if image_sizes is None:
        return [(int(images.shape[-2]), int(images.shape[-1]))] * int(images.shape[0])
    if torch.is_tensor(image_sizes):
        image_sizes = image_sizes.tolist()
    sizes = [(int(h), int(w)) for h, w in image_sizes]
    if len(sizes) != images.shape[0]:
        raise ValueError(f"Got {len(sizes)} image sizes for a batch of {images.shape[0]} images.")
    return sizes


def add_gt_boxes_to_rois(rois: torch.Tensor, gt_boxes: Sequence[torch.Tensor]) -> torch.Tensor:
    """Append every image's ground-truth boxes to the (R, 5) ROI list.

    Standard R-CNN training trick: early in training the RPN proposes little
    that overlaps the objects, so the GT boxes themselves guarantee the ROI
    head sees positive samples.
    """
    extra = [
        torch.cat((torch.full((boxes.shape[0], 1), float(i), device=rois.device, dtype=rois.dtype),
                   boxes.to(device=rois.device, dtype=rois.dtype)), dim=1)
        for i, boxes in enumerate(gt_boxes)
    ]
    return torch.cat([rois, *extra], dim=0)
