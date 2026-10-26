from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou

from ..base_matcher import Matcher


__all__ = [
    "HungarianMatcher", "HungarianMatcherConfig", "linear_assignment", "batched_linear_assignment",
    "batched_generalized_box_iou", "hungarian_matcher",
]


def linear_assignment(cost: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimum-cost one-to-one assignment of a (R, C) cost matrix (scipy's
    Hungarian solver). Returns (rows, cols) as long tensors; min(R, C) pairs."""
    if isinstance(cost, torch.Tensor):
        cost = cost.detach().float().cpu().numpy()
    if cost.size == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    rows, cols = linear_sum_assignment(cost)
    return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)


def batched_linear_assignment(cost: torch.Tensor, row_mask: torch.Tensor, col_counts: list[int]) -> torch.Tensor:
    """Hungarian assignment for a whole batch with a single device -> host transfer.

    :param cost: (B, R, C) cost, padded: image b uses rows ``row_mask[b]`` and its first
        ``col_counts[b]`` columns (whatever the padded entries hold is ignored).
    :param row_mask: (B, R) bool, the real rows (queries).
    :param col_counts: per image, the number of real columns (targets).
    :return: (B, R) long on ``cost.device``: the column assigned to each row, -1 = none.

    Matching image by image on the GPU costs a synchronisation per image (and per
    decoder layer); here the costs come to the CPU once and only the small scipy
    solves loop.
    """
    B, R, _ = cost.shape
    match = np.full((B, R), -1, dtype=np.int64)
    if sum(col_counts) > 0:
        cost_np = cost.detach().float().cpu().numpy()
        rows_np = row_mask.detach().cpu().numpy()
        for b, n in enumerate(col_counts):
            rows = np.flatnonzero(rows_np[b])
            if n == 0 or rows.size == 0:
                continue
            r, c = linear_sum_assignment(cost_np[b][rows][:, :n])
            match[b, rows[r]] = c
    return torch.from_numpy(match).to(cost.device, non_blocking=True)


def batched_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalised IoU of every pair, per image: (B, M, 4) and (B, G, 4) XYXY -> (B, M, G).
    Same values as torchvision's ``generalized_box_iou`` applied image by image."""
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    lt = torch.maximum(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
    rb = torch.minimum(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area1[:, :, None] + area2[:, None, :] - inter
    iou = inter / union
    lt_c = torch.minimum(boxes1[:, :, None, :2], boxes2[:, None, :, :2])
    rb_c = torch.maximum(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])
    enclosing = (rb_c - lt_c).clamp(min=0).prod(-1)
    return iou - (enclosing - union) / enclosing


@dataclass
class HungarianMatcherConfig:
    # boxes (DETR): class, L1 on normalised cxcywh, generalised IoU
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    # points: distance in units of point_scale pixels, minus objectness
    cost_point: float = 1.0
    cost_objectness: float = 1.0
    point_scale: float = 1.0


class HungarianMatcher(Matcher):
    """One-to-one matching of set predictions to ground truth (DETR's bipartite
    matching). No gradients flow through it.

    Unlike the IoU matcher it assigns every target to exactly one prediction,
    so it implements ``match_boxes`` / ``match_points`` (one image at a time) and
    their batched twins ``match_boxes_batched`` / ``match_points_batched`` (the
    criteria use these: one host transfer per decoder layer instead of one per
    image) instead of the anchor-labelling ``match``.
    """

    def __init__(self, cfg: HungarianMatcherConfig = HungarianMatcherConfig()):
        self.cfg = cfg

    def match(self, *args, **kwargs):
        raise NotImplementedError("HungarianMatcher matches sets: use match_boxes() or match_points().")

    @torch.no_grad()
    def box_cost(self, logits, boxes, gt_labels, gt_boxes) -> torch.Tensor:
        """(M, K+1) logits, (M, 4) and (G, 4) normalised cxcywh boxes -> (M, G) cost:
        -cost_class * p(class) + cost_bbox * L1 - cost_giou * GIoU."""
        prob = logits.float().softmax(-1)[:, gt_labels.long()]
        l1 = torch.cdist(boxes.float(), gt_boxes.float(), p=1)
        giou = generalized_box_iou(box_convert(boxes.float(), "cxcywh", "xyxy"), box_convert(gt_boxes.float(), "cxcywh", "xyxy"))
        return self.cfg.cost_bbox * l1 - self.cfg.cost_class * prob - self.cfg.cost_giou * giou

    @torch.no_grad()
    def point_cost(self, points, objectness, gt_points) -> torch.Tensor:
        """(M, 2) and (G, 2) pixel points, (M,) objectness logits -> (M, G) cost:
        cost_point * distance / point_scale - cost_objectness * sigmoid(objectness)."""
        dist = torch.cdist(points.float(), gt_points.float()) / self.cfg.point_scale
        return self.cfg.cost_point * dist - self.cfg.cost_objectness * objectness.float().sigmoid()[:, None]

    def match_boxes(self, logits, boxes, gt_labels, gt_boxes) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (prediction indices, target indices)."""
        return self._assign(self.box_cost(logits, boxes, gt_labels, gt_boxes), logits.device)

    def match_points(self, points, objectness, gt_points) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (prediction indices, target indices)."""
        return self._assign(self.point_cost(points, objectness, gt_points), points.device)

    @staticmethod
    def _assign(cost: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = linear_assignment(cost)
        return rows.to(device), cols.to(device)

    # --- batched: the whole batch at once (padded targets) ---------------------------------

    @torch.no_grad()
    def point_cost_batched(self, points, objectness, gt_points) -> torch.Tensor:
        """(B, M, 2) and (B, G, 2) pixel points, (B, M) objectness logits -> (B, M, G);
        the same cost as ``point_cost`` for every image."""
        dist = torch.cdist(points.float(), gt_points.float()) / self.cfg.point_scale
        return self.cfg.cost_point * dist - self.cfg.cost_objectness * objectness.float().sigmoid()[..., None]

    @torch.no_grad()
    def box_cost_batched(self, logits, boxes, gt_labels, gt_boxes) -> torch.Tensor:
        """(B, M, K+1) logits, (B, M, 4) and (B, G, 4) normalised cxcywh boxes, (B, G) labels
        -> (B, M, G); the same cost as ``box_cost`` for every image. Padded targets must hold a
        valid label (>= 0) and a non-degenerate box; their columns are ignored by the assignment."""
        M = logits.shape[1]
        prob = logits.float().softmax(-1).gather(2, gt_labels.long()[:, None, :].expand(-1, M, -1))
        l1 = torch.cdist(boxes.float(), gt_boxes.float(), p=1)
        giou = batched_generalized_box_iou(box_convert(boxes.float(), "cxcywh", "xyxy"),
                                           box_convert(gt_boxes.float(), "cxcywh", "xyxy"))
        return self.cfg.cost_bbox * l1 - self.cfg.cost_class * prob - self.cfg.cost_giou * giou

    def match_points_batched(self, points, objectness, gt_points, query_mask, counts) -> torch.Tensor:
        """-> (B, M) long: index of the target matched to each query, -1 = none."""
        return batched_linear_assignment(self.point_cost_batched(points, objectness, gt_points), query_mask, counts)

    def match_boxes_batched(self, logits, boxes, gt_labels, gt_boxes, query_mask, counts) -> torch.Tensor:
        """-> (B, M) long: index of the target matched to each query, -1 = none."""
        return batched_linear_assignment(self.box_cost_batched(logits, boxes, gt_labels, gt_boxes), query_mask, counts)


@Matcher.register(config=HungarianMatcherConfig())
def hungarian_matcher(cfg: HungarianMatcherConfig) -> HungarianMatcher:
    return HungarianMatcher(cfg)
