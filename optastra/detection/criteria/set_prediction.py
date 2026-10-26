"""Set-prediction losses (DETR family): Hungarian matching, then per-match losses.

* ``detr_criterion`` -- boxes + classes: cross-entropy (no-object class
  down-weighted by ``eos_coef``), L1 on normalised (cx, cy, w, h), generalised IoU.
* ``point_set_criterion`` -- points + objectness: L1 (or Laplace NLL) in units of
  ``loc_scale`` pixels and a sigmoid focal objectness loss.

Both are applied to every decoder layer (``extra["aux_outputs"]``) with
``aux_loss=True``; every layer is matched on its own, as in DETR. Losses are
normalised by the number of targets in the batch (per process).
``point_set_losses`` is the loss given an assignment, for callers that do
their own association (e.g. matching anchored queries by identity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torchvision.ops import box_convert

from ...core.component_ref import ComponentRef, ComponentRefConfigMixin, component_field
from ...losses.functional import sigmoid_focal_loss
from ...nn.features import HeadOutput
from .. import keys
from ..base_criterion import DetectionCriterion
from ..base_matcher import Matcher
from ..matching.hungarian import batched_generalized_box_iou


__all__ = [
    "DETRCriterion", "DETRCriterionConfig", "detr_criterion",
    "PointSetCriterion", "PointSetCriterionConfig", "point_set_criterion",
    "point_set_losses", "image_size_wh", "layer_outputs",
]


def image_size_wh(raw_preds: HeadOutput, batch_size: int, device) -> torch.Tensor:
    """(B, 2) per-image (w, h) from ``extra["image_sizes"]``."""
    sizes = raw_preds.extra.get(keys.IMAGE_SIZES)
    if sizes is None:
        raise ValueError("Set-prediction criteria need HeadOutput.extra['image_sizes'] (the DETR architecture sets it).")
    if torch.is_tensor(sizes):
        sizes = sizes.tolist()
    if len(sizes) != batch_size:
        raise ValueError(f"Got {len(sizes)} image sizes for a batch of {batch_size}.")
    return torch.tensor([[float(w), float(h)] for h, w in sizes], device=device)


def layer_outputs(raw_preds: HeadOutput, aux_loss: bool) -> list[tuple[str, HeadOutput]]:
    """[(suffix, output)]: the final layer ("") and, with aux_loss, every earlier one ("_aux{i}")."""
    out = [("", raw_preds)]
    if aux_loss:
        out += [(f"_aux{i}", aux) for i, aux in enumerate(raw_preds.extra.get(keys.AUX_OUTPUTS, []))]
    return out


def _query_mask(raw_preds: HeadOutput, shape) -> torch.Tensor:
    mask = raw_preds.extra.get(keys.QUERY_MASK)
    return mask if mask is not None else torch.ones(shape, dtype=torch.bool, device=raw_preds.scores.device
                                                    if raw_preds.scores is not None else raw_preds.logits.device)


# --- boxes ------------------------------------------------------------------------------

@dataclass
class DETRCriterionConfig(ComponentRefConfigMixin):
    num_classes: int = 80          # foreground classes (the DetectionTask injects its own)
    matcher: ComponentRef = component_field(Matcher, default_name="hungarian_matcher")
    eos_coef: float = 0.1          # weight of the "no object" class in the cross-entropy
    loss_ce: float = 1.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    aux_loss: bool = True


class DETRCriterion(DetectionCriterion):
    """DETR's set loss (Carion et al., 2020). Targets: per image ``boxes`` (G, 4)
    XYXY absolute pixels and ``labels`` (G,) in [0, num_classes)."""

    required_fields = ("logits", "boxes")

    def __init__(self, cfg: DETRCriterionConfig):
        self.cfg = cfg
        self.matcher = cfg.matcher.resolve(Matcher)

    def validate_predictions(self, raw_preds: HeadOutput) -> None:
        super().validate_predictions(raw_preds)
        if raw_preds.logits.shape[-1] != self.cfg.num_classes + 1:
            raise ValueError(f"detr_criterion expects {self.cfg.num_classes} + 1 class logits (last = no object), "
                             f"got {raw_preds.logits.shape[-1]}.")

    def _targets(self, raw_preds, targets):
        B = raw_preds.logits.shape[0]
        wh = image_size_wh(raw_preds, B, raw_preds.logits.device)
        scale = torch.cat([wh, wh], dim=1)                                           # (B, 4)
        device = raw_preds.logits.device
        per_image = [(t["labels"].long().reshape(-1).to(device),
                      box_convert(t["boxes"].float().reshape(-1, 4).to(device), "xyxy", "cxcywh") / scale[b])
                     for b, t in enumerate(targets)]
        counts = [int(labels.numel()) for labels, _ in per_image]
        G = max(max(counts, default=0), 1)   # >= 1 column, so the gathers below work without targets
        # padded targets: label 0 and a unit-centred box keep the cost finite; the assignment ignores them
        labels = torch.zeros(B, G, dtype=torch.long, device=device)
        boxes = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device).repeat(B, G, 1)
        for b, (l, bx) in enumerate(per_image):
            labels[b, : l.numel()], boxes[b, : l.numel()] = l, bx
        return labels, boxes, counts

    def _match(self, out: HeadOutput, labels, boxes, counts, qmask) -> torch.Tensor:
        """(B, M) index of each query's target (-1 = none); one host transfer for the batch."""
        return self.matcher.match_boxes_batched(out.logits, out.boxes, labels, boxes, qmask, counts)

    def _layer_losses(self, out: HeadOutput, labels, boxes, counts, qmask, num_boxes):
        K = self.cfg.num_classes
        match = self._match(out, labels, boxes, counts, qmask)
        sel = match >= 0
        idx = match.clamp_min(0)
        target_classes = torch.where(sel, torch.gather(labels, 1, idx), torch.full_like(match, K))
        weight = torch.ones(K + 1, device=out.logits.device)
        weight[K] = self.cfg.eos_coef
        ce = F.cross_entropy(out.logits[qmask].float(), target_classes[qmask], weight=weight)
        pred = out.boxes[sel].float()
        true = torch.gather(boxes, 1, idx[..., None].expand(-1, -1, 4))[sel]
        l1 = (pred - true).abs().sum() / num_boxes
        # GIoU of each matched (prediction, target) pair: the pairwise formula on (P, 1, 4) boxes
        giou = batched_generalized_box_iou(box_convert(pred, "cxcywh", "xyxy")[:, None],
                                           box_convert(true, "cxcywh", "xyxy")[:, None]).reshape(-1)
        return {
            "loss_ce": self.cfg.loss_ce * ce,
            "loss_bbox": self.cfg.loss_bbox * l1,
            "loss_giou": self.cfg.loss_giou * (1 - giou).sum() / num_boxes,
        }, match, target_classes

    def compute_losses(self, raw_preds: HeadOutput, targets: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels, boxes, counts = self._targets(raw_preds, targets)
        qmask = _query_mask(raw_preds, raw_preds.logits.shape[:2])
        num_boxes = max(sum(counts), 1)
        losses = {}
        for suffix, out in layer_outputs(raw_preds, self.cfg.aux_loss):
            layer, _, _ = self._layer_losses(out, labels, boxes, counts, qmask, num_boxes)
            losses.update({k + suffix: v for k, v in layer.items()})
        return losses

    @torch.no_grad()
    def compute_metrics(self, raw_preds: HeadOutput, targets) -> dict[str, float]:
        labels, boxes, counts = self._targets(raw_preds, targets)
        qmask = _query_mask(raw_preds, raw_preds.logits.shape[:2])
        match = self._match(raw_preds, labels, boxes, counts, qmask)
        sel = match >= 0
        if not sel.any():
            return {}
        pred = raw_preds.logits.argmax(-1)[sel]
        true = torch.gather(labels, 1, match.clamp_min(0))[sel]
        return {"matched_class_accuracy": float((pred == true).float().mean())}


# --- points -----------------------------------------------------------------------------

def point_set_losses(
    points: torch.Tensor,
    objectness: torch.Tensor,
    match: torch.Tensor,
    query_mask: torch.Tensor,
    gt_points: torch.Tensor,
    *,
    log_scales: torch.Tensor | None = None,
    loc_loss: str = "l1",
    loc_scale: float = 1.0,
    loc_weight: float = 5.0,
    objectness_weight: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    noobj_weight: float = 1.0,
    num_targets: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Localisation and objectness losses for a given assignment.

    :param points: (B, M, 2) predicted points, pixels.
    :param objectness: (B, M) logits.
    :param match: (B, M) long, index into ``gt_points`` of each query's target, -1 = none.
    :param query_mask: (B, M) bool, real queries.
    :param gt_points: (B, G, 2) target points, pixels.
    :param log_scales: (B, M, 2) log of the predicted scale in pixels (``loc_loss="laplace_nll"``).
    :param num_targets: normaliser; default = number of matched queries (at least 1).
    :return: (loc, obj), already multiplied by their weights. Matched queries regress
        their target and are objectness positives; other real queries are no-object
        negatives weighted by ``noobj_weight``. Distances are in units of ``loc_scale`` px.
    """
    if loc_loss not in ("l1", "laplace_nll"):
        raise ValueError(f"loc_loss must be 'l1' or 'laplace_nll', got {loc_loss!r}.")
    matched = (match >= 0) & query_mask
    num = torch.as_tensor(float(num_targets) if num_targets is not None else float(matched.sum()),
                          device=points.device).clamp_min(1.0)
    idx = match.clamp_min(0)
    if gt_points.shape[1] == 0:
        target = torch.zeros_like(points)
    else:
        target = torch.gather(gt_points.float(), 1, idx[..., None].expand(-1, -1, 2))
    delta = (points.float() - target)[matched] / loc_scale
    if loc_loss == "l1":
        loc = delta.abs().sum() / num
    else:
        if log_scales is None:
            raise ValueError("loc_loss='laplace_nll' needs predicted log scales.")
        ls = log_scales[matched].float() - torch.log(torch.tensor(loc_scale, device=points.device))
        loc = (delta.abs() * torch.exp(-ls) + ls).sum() / num                   # Laplace NLL, log 2 dropped
    focal = sigmoid_focal_loss(objectness.float(), matched.float(), focal_alpha, focal_gamma)
    w = torch.where(matched, torch.ones_like(focal), torch.full_like(focal, noobj_weight))
    obj = (focal * w * query_mask.float()).sum() / num
    return loc_weight * loc, objectness_weight * obj


@dataclass
class PointSetCriterionConfig(ComponentRefConfigMixin):
    matcher: ComponentRef = component_field(Matcher, default_name="hungarian_matcher")
    loc_loss: str = "l1"           # "l1" | "laplace_nll" (keypoint_head with predict_uncertainty=True)
    loc_scale: float = 1.0         # pixels
    loc_weight: float = 5.0
    objectness_weight: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    noobj_weight: float = 1.0
    aux_loss: bool = True


class PointSetCriterion(DetectionCriterion):
    """Set loss for point detection (keypoint_head). Targets: per image
    ``points`` (G, 2) pixel (x, y)."""

    required_fields = ("scores",)

    def __init__(self, cfg: PointSetCriterionConfig):
        if cfg.loc_loss not in ("l1", "laplace_nll"):
            raise ValueError(f"loc_loss must be 'l1' or 'laplace_nll', got {cfg.loc_loss!r}.")
        self.cfg = cfg
        self.matcher = cfg.matcher.resolve(Matcher)

    def validate_predictions(self, raw_preds: HeadOutput) -> None:
        super().validate_predictions(raw_preds)
        if keys.POINTS not in raw_preds.extra:
            raise ValueError("point_set_criterion needs HeadOutput.extra['points'] (use keypoint_head).")
        if self.cfg.loc_loss == "laplace_nll" and keys.POINT_LOG_SCALES not in raw_preds.extra:
            raise ValueError("loc_loss='laplace_nll' needs keypoint_head(predict_uncertainty=True).")

    @staticmethod
    def pixels(out: HeadOutput, wh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Points and log scales of one layer in pixels."""
        points = out.extra[keys.POINTS].float() * wh[:, None, :]
        log_scales = out.extra.get(keys.POINT_LOG_SCALES)
        if log_scales is not None:
            log_scales = log_scales.float() + torch.log(wh)[:, None, :]
        return points, log_scales

    def _gt(self, targets, device) -> tuple[torch.Tensor, list[int]]:
        pts = [t["points"].float().reshape(-1, 2).to(device) for t in targets]
        G = max((p.shape[0] for p in pts), default=0)
        gt = torch.zeros(len(pts), G, 2, device=device)
        for b, p in enumerate(pts):
            gt[b, : p.shape[0]] = p
        return gt, [p.shape[0] for p in pts]

    def _match(self, points, objectness, qmask, gt, counts) -> torch.Tensor:
        """(B, M) index of each query's target (-1 = none); one host transfer for the batch."""
        return self.matcher.match_points_batched(points, objectness, gt, qmask, counts)

    def _layer_losses(self, out: HeadOutput, points, log_scales, match, qmask, gt, counts, targets) -> dict[str, torch.Tensor]:
        """Losses of one decoder layer given its matching. Subclasses add terms that use the
        same ``match`` (e.g. a value regressed for each matched query) by extending this."""
        c = self.cfg
        loc, obj = point_set_losses(
            points, out.scores, match, qmask, gt, log_scales=log_scales, loc_loss=c.loc_loss,
            loc_scale=c.loc_scale, loc_weight=c.loc_weight, objectness_weight=c.objectness_weight,
            focal_alpha=c.focal_alpha, focal_gamma=c.focal_gamma, noobj_weight=c.noobj_weight,
            num_targets=sum(counts),
        )
        return {"loss_loc": loc, "loss_objectness": obj}

    def compute_losses(self, raw_preds: HeadOutput, targets) -> dict[str, torch.Tensor]:
        B = raw_preds.scores.shape[0]
        wh = image_size_wh(raw_preds, B, raw_preds.scores.device)
        gt, counts = self._gt(targets, raw_preds.scores.device)
        qmask = _query_mask(raw_preds, raw_preds.scores.shape)
        losses = {}
        for suffix, out in layer_outputs(raw_preds, self.cfg.aux_loss):
            points, log_scales = self.pixels(out, wh)
            match = self._match(points, out.scores, qmask, gt, counts)
            layer = self._layer_losses(out, points, log_scales, match, qmask, gt, counts, targets)
            losses.update({k + suffix: v for k, v in layer.items()})
        return losses

    @torch.no_grad()
    def compute_metrics(self, raw_preds: HeadOutput, targets) -> dict[str, float]:
        B = raw_preds.scores.shape[0]
        wh = image_size_wh(raw_preds, B, raw_preds.scores.device)
        gt, counts = self._gt(targets, raw_preds.scores.device)
        qmask = _query_mask(raw_preds, raw_preds.scores.shape)
        points, _ = self.pixels(raw_preds, wh)
        match = self._match(points, raw_preds.scores, qmask, gt, counts)
        sel = match >= 0
        if not sel.any():
            return {}
        target = torch.gather(gt, 1, match.clamp_min(0)[..., None].expand(-1, -1, 2))
        return {"loc_error_px": float((points - target).norm(dim=-1)[sel].mean())}


@DetectionCriterion.register(config=DETRCriterionConfig())
def detr_criterion(cfg: DETRCriterionConfig) -> DETRCriterion:
    return DETRCriterion(cfg)


@DetectionCriterion.register(config=PointSetCriterionConfig())
def point_set_criterion(cfg: PointSetCriterionConfig) -> PointSetCriterion:
    return PointSetCriterion(cfg)
