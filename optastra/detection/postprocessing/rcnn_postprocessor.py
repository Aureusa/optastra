from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...data.sample import Sample
from ...nn.blocks.geometry.boxes import apply_deltas_to_anchors, batched_nms, clip_boxes_to_image
from ...nn.features import HeadOutput
from .. import keys
from ..base_postprocessor import Postprocessor


__all__ = ["RCNNPostprocessor", "RCNNPostprocessorConfig", "paste_masks_in_image"]


@dataclass
class RCNNPostprocessorConfig:
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_image: int = 100
    bbox_reg_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    mask_threshold: float | None = 0.5  # None keeps the soft (sigmoid) mask probabilities


def paste_masks_in_image(
    mask_probs: torch.Tensor,
    boxes: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Resize each fixed-size ROI mask to its box and paste it into an image-size canvas.

    :param mask_probs: (K, M, M) mask probabilities, one per box.
    :param boxes: (K, 4) XYXY boxes (image coordinates) the masks were predicted for.
    :param image_size: (h, w) of the output masks.
    :return: (K, h, w) masks, zero outside each box.
    """
    height, width = image_size
    out = mask_probs.new_zeros((mask_probs.shape[0], height, width))
    for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        # Integer pixel extent covered by the box (at least one pixel).
        left, top = math.floor(x1), math.floor(y1)
        right, bottom = max(math.ceil(x2), left + 1), max(math.ceil(y2), top + 1)
        resized = F.interpolate(
            mask_probs[i][None, None], size=(bottom - top, right - left), mode="bilinear", align_corners=False
        )[0, 0]

        # Only the part of the box that lies inside the image is pasted.
        x0, y0 = max(left, 0), max(top, 0)
        x1_in, y1_in = min(right, width), min(bottom, height)
        if x1_in <= x0 or y1_in <= y0:
            continue
        out[i, y0:y1_in, x0:x1_in] = resized[y0 - top : y1_in - top, x0 - left : x1_in - left]
    return out


class RCNNPostprocessor(Postprocessor):
    """Decode ROI head outputs into one ``Sample`` per image.

    Per image: softmax the class logits, decode every class's box with that
    class's own deltas (or the shared deltas for class-agnostic regression),
    clip to the image's own unpadded size, then score-threshold + per-class NMS
    and keep the top ``detections_per_image``.

    Masks (if predicted): sigmoid, take the channel of the detected class and
    paste it back into an image-size mask at the ROI box the mask head saw.
    """

    def __init__(self, cfg: RCNNPostprocessorConfig):
        self.cfg = cfg

    def _decode_boxes(self, deltas: torch.Tensor, proposals: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Return (R, num_classes, 4) boxes: row r, column c is proposal r refined as class c."""
        if deltas.shape[1] == 4:
            boxes = apply_deltas_to_anchors(deltas, proposals, weights=self.cfg.bbox_reg_weights)
            return boxes[:, None, :].expand(-1, num_classes, 4)
        if deltas.shape[1] != 4 * num_classes:
            raise ValueError(f"Expected 4 or {4 * num_classes} box deltas per ROI, got {deltas.shape[1]}.")
        per_class_deltas = deltas.view(deltas.shape[0], num_classes, 4)
        return apply_deltas_to_anchors(
            per_class_deltas, proposals[:, None, :].expand_as(per_class_deltas), weights=self.cfg.bbox_reg_weights
        )

    def process(self, raw_preds: HeadOutput, num_classes: int) -> list[Sample]:
        if raw_preds.logits is None or raw_preds.values is None:
            return []
        roi_boxes = raw_preds.extra.get(keys.ROI_BOXES)
        image_sizes = raw_preds.extra.get(keys.IMAGE_SIZES)
        if roi_boxes is None or image_sizes is None:
            raise ValueError(
                f"RCNNPostprocessor needs raw_preds.extra['{keys.ROI_BOXES}'] and "
                f"raw_preds.extra['{keys.IMAGE_SIZES}'] (one (h, w) per image in the batch)."
            )
        if raw_preds.logits.shape[1] != num_classes + 1:
            raise ValueError(
                f"Expected num_classes + 1 = {num_classes + 1} class logits, got {raw_preds.logits.shape[1]}."
            )

        scores = raw_preds.logits.float().softmax(dim=1)[:, :-1]  # (R, C), background dropped
        proposals = roi_boxes[:, 1:].float()
        boxes = self._decode_boxes(raw_preds.values.float(), proposals, num_classes)  # (R, C, 4)
        batch_ids = roi_boxes[:, 0].long()

        out: list[Sample] = []
        for image_index, image_size in enumerate(image_sizes):
            image_size = (int(image_size[0]), int(image_size[1]))
            rois = torch.where(batch_ids == image_index)[0]
            boxes_i = clip_boxes_to_image(boxes[rois], image_size)
            scores_i = scores[rois]

            # Every (roi, class) pair above the threshold is a candidate detection.
            roi_idx, class_idx = torch.nonzero(scores_i >= self.cfg.score_thresh, as_tuple=True)
            cand_boxes = boxes_i[roi_idx, class_idx]
            cand_scores = scores_i[roi_idx, class_idx]
            keep = batched_nms(cand_boxes, cand_scores, class_idx, self.cfg.nms_thresh)
            keep = keep[: self.cfg.detections_per_image]  # batched_nms sorts by score

            detections = {
                "boxes": cand_boxes[keep],
                "scores": cand_scores[keep],
                "labels": class_idx[keep],
            }
            if raw_preds.masks is not None:
                detections["masks"] = self._decode_masks(
                    raw_preds.masks[rois[roi_idx[keep]]], class_idx[keep], proposals[rois[roi_idx[keep]]], image_size
                )
            out.append(Sample(target=detections, meta={"image_index": image_index, "image_size": image_size}))

        return out

    def _decode_masks(
        self,
        mask_logits: torch.Tensor,
        labels: torch.Tensor,
        roi_boxes: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """(K, C or 1, M, M) logits -> (K, h, w) masks for the detected classes."""
        channel = labels if mask_logits.shape[1] > 1 else torch.zeros_like(labels)
        probs = mask_logits[torch.arange(mask_logits.shape[0], device=mask_logits.device), channel].float().sigmoid()
        masks = paste_masks_in_image(probs, roi_boxes, image_size)
        if self.cfg.mask_threshold is not None:
            masks = masks >= self.cfg.mask_threshold
        return masks


postprocessor_configs = {
    "rcnn_postprocessor": RCNNPostprocessorConfig(),
}


@Postprocessor.register(config=postprocessor_configs["rcnn_postprocessor"])
def rcnn_postprocessor(cfg: RCNNPostprocessorConfig) -> RCNNPostprocessor:
    return RCNNPostprocessor(cfg)
