from __future__ import annotations

from dataclasses import dataclass

import torch

from .._registry import register_postprocessor
from ...data.sample import Sample
from ...nn.blocks.geometry.boxes import apply_deltas_to_anchors, batched_nms, clip_boxes_to_image
from ...nn.features import HeadOutput


@dataclass
class RCNNPostprocessorConfig:
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_image: int = 100
    bbox_reg_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


class RCNNPostprocessor:
    """Decode RoI head outputs into per-image Samples with class-wise NMS."""

    def __init__(self, cfg: RCNNPostprocessorConfig):
        self.cfg = cfg

    def process(self, raw_preds: HeadOutput, num_classes: int, image_size: tuple[int, int] | None = None) -> list[Sample]:
        if raw_preds.logits is None or raw_preds.values is None:
            return []
        roi_boxes = raw_preds.extra.get("roi_boxes")
        if roi_boxes is None:
            return []

        image_size = image_size or (0, 0)
        scores = raw_preds.logits.softmax(dim=1)
        fg_scores = scores[:, :-1]

        box_deltas = raw_preds.values
        if box_deltas.shape[1] == 4:
            cls_idx = fg_scores.argmax(dim=1)
            picked_deltas = box_deltas
        else:
            cls_idx = fg_scores.argmax(dim=1)
            all_deltas = box_deltas.view(box_deltas.shape[0], num_classes, 4)
            picked_deltas = all_deltas[torch.arange(all_deltas.shape[0], device=all_deltas.device), cls_idx]

        decoded = apply_deltas_to_anchors(picked_deltas, roi_boxes[:, 1:], weights=self.cfg.bbox_reg_weights)
        if image_size[0] > 0 and image_size[1] > 0:
            decoded = clip_boxes_to_image(decoded, image_size)

        batch_ids = roi_boxes[:, 0].long()
        out: list[Sample] = []
        num_images = int(batch_ids.max().item()) + 1 if batch_ids.numel() > 0 else 0

        for image_idx in range(num_images):
            keep_img = torch.where(batch_ids == image_idx)[0]
            if keep_img.numel() == 0:
                out.append(Sample(target={}, meta={"image_index": image_idx, "image_size": image_size}))
                continue

            boxes_i = decoded[keep_img]
            scores_i = fg_scores[keep_img]

            selected_boxes: list[torch.Tensor] = []
            selected_scores: list[torch.Tensor] = []
            selected_labels: list[torch.Tensor] = []
            selected_idx: list[torch.Tensor] = []

            for cls in range(num_classes):
                cls_score = scores_i[:, cls]
                valid = torch.where(cls_score >= self.cfg.score_thresh)[0]
                if valid.numel() == 0:
                    continue
                boxes_c = boxes_i[valid]
                scores_c = cls_score[valid]
                level_ids = torch.zeros_like(scores_c, dtype=torch.long)
                keep = batched_nms(boxes_c, scores_c, level_ids, self.cfg.nms_thresh)

                selected_boxes.append(boxes_c[keep])
                selected_scores.append(scores_c[keep])
                selected_labels.append(torch.full((keep.numel(),), cls, dtype=torch.long, device=boxes_i.device))
                selected_idx.append(keep_img[valid[keep]])

            if not selected_boxes:
                out.append(Sample(target={}, meta={"image_index": image_idx, "image_size": image_size}))
                continue

            boxes_cat = torch.cat(selected_boxes, dim=0)
            scores_cat = torch.cat(selected_scores, dim=0)
            labels_cat = torch.cat(selected_labels, dim=0)
            idx_cat = torch.cat(selected_idx, dim=0)

            if scores_cat.numel() > self.cfg.detections_per_image:
                topk = torch.topk(scores_cat, k=self.cfg.detections_per_image).indices
                boxes_cat = boxes_cat[topk]
                scores_cat = scores_cat[topk]
                labels_cat = labels_cat[topk]
                idx_cat = idx_cat[topk]

            pred_target = {
                "boxes": boxes_cat,
                "scores": scores_cat,
                "labels": labels_cat,
            }
            if raw_preds.masks is not None:
                pred_target["masks"] = raw_preds.masks[idx_cat]
            out.append(Sample(target=pred_target, meta={"image_index": image_idx, "image_size": image_size}))

        return out


postprocessor_configs = {
    "rcnn_postprocessor": RCNNPostprocessorConfig(),
}


@register_postprocessor(config=postprocessor_configs["rcnn_postprocessor"])
def rcnn_postprocessor(cfg: RCNNPostprocessorConfig) -> RCNNPostprocessor:
    return RCNNPostprocessor(cfg)
