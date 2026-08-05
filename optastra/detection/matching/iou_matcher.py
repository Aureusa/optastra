from __future__ import annotations

from dataclasses import dataclass

import torch

from .._registry import register_matcher
from ...nn.blocks.geometry.boxes import pairwise_iou


@dataclass
class IoUMatcherConfig:
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.5
    allow_low_quality_matches: bool = True


class IoUMatcher:
    """Match proposals/anchors to GT using IoU thresholds."""

    def __init__(self, cfg: IoUMatcherConfig):
        self.cfg = cfg

    def match(
        self,
        proposals: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        *,
        background_label: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (labels, matched_gt_idx).

        labels are in {class_id, background_label, -1(ignore)}.
        """
        device = proposals.device
        if gt_boxes.numel() == 0:
            labels = torch.full((proposals.shape[0],), background_label, device=device, dtype=torch.long)
            matched = torch.zeros((proposals.shape[0],), device=device, dtype=torch.long)
            return labels, matched

        ious = pairwise_iou(proposals, gt_boxes)
        matched_iou, matched = ious.max(dim=1)

        labels = torch.full((proposals.shape[0],), -1, device=device, dtype=torch.long)
        fg = matched_iou >= self.cfg.fg_iou_thresh
        bg = matched_iou < self.cfg.bg_iou_thresh

        labels[fg] = gt_labels[matched[fg]].long()
        labels[bg] = background_label

        if self.cfg.allow_low_quality_matches:
            # Ensure every GT has at least one positive match.
            best_per_gt_iou, best_per_gt_idx = ious.max(dim=0)
            keep = best_per_gt_iou > 0
            if keep.any():
                chosen = best_per_gt_idx[keep]
                gt_idx = torch.where(keep)[0]
                labels[chosen] = gt_labels[gt_idx].long()
                matched[chosen] = gt_idx.long()

        return labels, matched


matcher_configs = {
    "iou_matcher": IoUMatcherConfig(),
    "rpn_iou_matcher": IoUMatcherConfig(fg_iou_thresh=0.7, bg_iou_thresh=0.3, allow_low_quality_matches=True),
}


@register_matcher(config=matcher_configs["iou_matcher"])
def iou_matcher(cfg: IoUMatcherConfig) -> IoUMatcher:
    return IoUMatcher(cfg)


@register_matcher(config=matcher_configs["rpn_iou_matcher"])
def rpn_iou_matcher(cfg: IoUMatcherConfig) -> IoUMatcher:
    return IoUMatcher(cfg)
