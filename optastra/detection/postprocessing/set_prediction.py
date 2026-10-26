from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import box_convert

from ...data.sample import Sample
from ...nn.blocks.geometry.boxes import clip_boxes_to_image
from ...nn.features import HeadOutput
from .. import keys
from ..base_postprocessor import Postprocessor


__all__ = [
    "DETRPostprocessor", "DETRPostprocessorConfig", "detr_postprocessor",
    "PointPostprocessor", "PointPostprocessorConfig", "point_postprocessor",
]


def _image_sizes(raw_preds: HeadOutput) -> list[tuple[int, int]]:
    sizes = raw_preds.extra.get(keys.IMAGE_SIZES)
    if sizes is None:
        raise ValueError("Set-prediction postprocessors need HeadOutput.extra['image_sizes'].")
    if torch.is_tensor(sizes):
        sizes = sizes.tolist()
    return [(int(h), int(w)) for h, w in sizes]


def _query_mask(raw_preds: HeadOutput, shape, device) -> torch.Tensor:
    mask = raw_preds.extra.get(keys.QUERY_MASK)
    return mask if mask is not None else torch.ones(shape, dtype=torch.bool, device=device)


@dataclass
class DETRPostprocessorConfig:
    score_thresh: float = 0.0
    detections_per_image: int = 100


class DETRPostprocessor(Postprocessor):
    """Set predictions -> one ``Sample`` per image with ``boxes`` (K, 4) XYXY pixels,
    ``scores`` and ``labels``: each query's best non-"no object" class. No NMS --
    the set loss already trains queries not to duplicate each other."""

    def __init__(self, cfg: DETRPostprocessorConfig = DETRPostprocessorConfig()):
        self.cfg = cfg

    @torch.no_grad()
    def process(self, raw_preds: HeadOutput, num_classes: int) -> list[Sample]:
        prob = raw_preds.logits.float().softmax(-1)[..., :-1]                    # drop "no object"
        scores, labels = prob.max(-1)
        qmask = _query_mask(raw_preds, scores.shape, scores.device)
        out = []
        for b, (h, w) in enumerate(_image_sizes(raw_preds)):
            boxes = box_convert(raw_preds.boxes[b].float(), "cxcywh", "xyxy") * scores.new_tensor([w, h, w, h])
            keep = torch.nonzero(qmask[b] & (scores[b] > self.cfg.score_thresh)).flatten()
            keep = keep[scores[b, keep].argsort(descending=True)][: self.cfg.detections_per_image]
            detections = {"boxes": clip_boxes_to_image(boxes[keep], (h, w)), "scores": scores[b, keep], "labels": labels[b, keep]}
            out.append(Sample(target=detections, meta={"image_index": b, "image_size": (h, w)}))
        return out


@dataclass
class PointPostprocessorConfig:
    score_thresh: float = 0.5
    max_points: int = 1000


class PointPostprocessor(Postprocessor):
    """Point predictions -> one ``Sample`` per image with ``points`` (K, 2) pixel
    (x, y) and ``scores`` (objectness probabilities) above ``score_thresh``."""

    def __init__(self, cfg: PointPostprocessorConfig = PointPostprocessorConfig()):
        self.cfg = cfg

    @torch.no_grad()
    def process(self, raw_preds: HeadOutput, num_classes: int | None = None) -> list[Sample]:
        scores = raw_preds.scores.float().sigmoid()
        qmask = _query_mask(raw_preds, scores.shape, scores.device)
        out = []
        for b, (h, w) in enumerate(_image_sizes(raw_preds)):
            points = raw_preds.extra[keys.POINTS][b].float() * scores.new_tensor([w, h])
            keep = torch.nonzero(qmask[b] & (scores[b] > self.cfg.score_thresh)).flatten()
            keep = keep[scores[b, keep].argsort(descending=True)][: self.cfg.max_points]
            out.append(Sample(target={"points": points[keep], "scores": scores[b, keep]},
                              meta={"image_index": b, "image_size": (h, w)}))
        return out


@Postprocessor.register(config=DETRPostprocessorConfig())
def detr_postprocessor(cfg: DETRPostprocessorConfig) -> DETRPostprocessor:
    return DETRPostprocessor(cfg)


@Postprocessor.register(config=PointPostprocessorConfig())
def point_postprocessor(cfg: PointPostprocessorConfig) -> PointPostprocessor:
    return PointPostprocessor(cfg)
