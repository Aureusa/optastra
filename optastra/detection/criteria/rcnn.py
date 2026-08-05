from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from ...core.component_ref import ComponentRef
from ..base_matcher import Matcher
from ..base_sampler import Sampler
from .._registry import register_criterion
from ..base_criterion import DetectionCriterion
from ...nn.blocks.geometry.boxes import encode_boxes
from ...nn.features import FeatureMaps, HeadOutput


def _labels_to_binary(labels: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    out = torch.full_like(labels, -1)
    out[(labels >= 0) & (labels < num_classes)] = 1
    out[labels == num_classes] = 0
    return out


@dataclass
class RCNNCriterionConfig:
    num_classes: int = 80
    bbox_reg_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    box_loss_beta: float = 1.0
    mask_loss_weight: float = 1.0
    roi_matcher: ComponentRef = field(default_factory=lambda: ComponentRef("iou_matcher"))
    roi_sampler: ComponentRef = field(default_factory=lambda: ComponentRef("rcnn_balanced_sampler"))
    rpn_matcher: ComponentRef = field(default_factory=lambda: ComponentRef("rpn_iou_matcher"))
    rpn_sampler: ComponentRef = field(default_factory=lambda: ComponentRef("rpn_balanced_sampler"))


class RCNNCriterion(DetectionCriterion):
    required_fields = ("logits", "values")

    def __init__(self, cfg: RCNNCriterionConfig):
        self.cfg = cfg
        self.roi_matcher = cfg.roi_matcher.resolve(Matcher)
        self.roi_sampler = cfg.roi_sampler.resolve(Sampler)
        self.rpn_matcher = cfg.rpn_matcher.resolve(Matcher)
        self.rpn_sampler = cfg.rpn_sampler.resolve(Sampler)

    def validate_predictions(self, raw_preds: HeadOutput) -> None:
        super().validate_predictions(raw_preds)
        if "roi_boxes" not in raw_preds.extra:
            raise ValueError("RCNNCriterion requires raw_preds.extra['roi_boxes'] from architecture forward().")

    def _box_reg_loss(
        self,
        pred_deltas: torch.Tensor,
        proposals: torch.Tensor,
        gt_boxes: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> torch.Tensor:
        fg = cls_targets < self.cfg.num_classes
        if fg.sum() == 0:
            return pred_deltas.sum() * 0.0

        fg_deltas = pred_deltas[fg]
        fg_targets = cls_targets[fg]
        target_deltas = encode_boxes(proposals[fg], gt_boxes[fg], weights=self.cfg.bbox_reg_weights)

        if fg_deltas.shape[1] == 4:
            pred = fg_deltas
        else:
            pred = fg_deltas.view(fg_deltas.shape[0], self.cfg.num_classes, 4)
            pred = pred[torch.arange(pred.shape[0], device=pred.device), fg_targets]

        return F.smooth_l1_loss(pred, target_deltas, beta=self.cfg.box_loss_beta, reduction="sum") / max(int(fg.sum().item()), 1)

    def _rpn_losses(self, rpn_output: FeatureMaps, targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not rpn_output.extra:
            zero = torch.tensor(0.0)
            return {"rpn_objectness_loss": zero, "rpn_box_loss": zero}

        objectness_by_level = rpn_output.extra["objectness_logits"]
        deltas_by_level = rpn_output.extra["bbox_deltas"]
        anchors_by_level = rpn_output.extra["anchors"]

        any_level = next(iter(objectness_by_level.values()))
        device = any_level.device
        total_obj = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)
        num_images = any_level.shape[0]

        for image_index in range(num_images):
            gt_boxes = targets[image_index]["boxes"]
            gt_labels = torch.ones((gt_boxes.shape[0],), dtype=torch.long, device=device)
            for level_name, obj_logits in objectness_by_level.items():
                level_logits = obj_logits[image_index].reshape(-1)
                _, anchors_per_loc, h, w = obj_logits.shape
                level_deltas = deltas_by_level[level_name][image_index].view(anchors_per_loc, 4, h, w)
                level_deltas = level_deltas.permute(2, 3, 0, 1).reshape(-1, 4)
                anchors = anchors_by_level[level_name]

                labels, matched_gt = self.rpn_matcher.match(
                    anchors,
                    gt_boxes,
                    gt_labels,
                    background_label=0,
                )
                sampled = self.rpn_sampler.sample(labels, positive_value=1, negative_value=0)
                if sampled.numel() == 0:
                    continue

                sampled_labels = labels[sampled].float()
                total_obj = total_obj + F.binary_cross_entropy_with_logits(level_logits[sampled], sampled_labels)

                pos = sampled[labels[sampled] == 1]
                if pos.numel() > 0 and gt_boxes.numel() > 0:
                    pred_pos = level_deltas[pos]
                    target_pos = encode_boxes(anchors[pos], gt_boxes[matched_gt[pos]], weights=self.cfg.bbox_reg_weights)
                    total_box = total_box + F.smooth_l1_loss(pred_pos, target_pos, beta=self.cfg.box_loss_beta, reduction="sum") / max(int(pos.numel()), 1)

        return {
            "rpn_objectness_loss": total_obj / max(num_images, 1),
            "rpn_box_loss": total_box / max(num_images, 1),
        }

    def _mask_loss(
        self,
        pred_masks: torch.Tensor,
        sampled_global_indices: torch.Tensor,
        cls_targets_all: torch.Tensor,
        matched_gt_global: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        roi_boxes: torch.Tensor,
    ) -> torch.Tensor:
        fg = cls_targets_all < self.cfg.num_classes
        fg_global = sampled_global_indices[fg]
        fg_classes = cls_targets_all[fg]
        if fg_global.numel() == 0:
            return pred_masks.sum() * 0.0

        h, w = pred_masks.shape[-2:]
        aligned_targets: list[torch.Tensor] = []
        pred_selected: list[torch.Tensor] = []

        for idx, class_idx in zip(fg_global.tolist(), fg_classes.tolist()):
            batch_idx = int(roi_boxes[idx, 0].item())
            if "masks" not in targets[batch_idx]:
                continue
            gt_idx = int(matched_gt_global[idx].item())
            gt_mask = targets[batch_idx]["masks"][gt_idx].float().unsqueeze(0).unsqueeze(0)
            roi = roi_boxes[idx : idx + 1].clone()
            roi[:, 0] = 0.0
            target_mask = roi_align(gt_mask, roi, output_size=(h, w), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
            target_mask = target_mask.squeeze(0).squeeze(0)

            if pred_masks.shape[1] == 1:
                pred_mask = pred_masks[idx, 0]
            else:
                pred_mask = pred_masks[idx, int(class_idx)]

            aligned_targets.append(target_mask)
            pred_selected.append(pred_mask)

        if not pred_selected:
            return pred_masks.sum() * 0.0

        pred_tensor = torch.stack(pred_selected, dim=0)
        target_tensor = torch.stack(aligned_targets, dim=0)
        return F.binary_cross_entropy_with_logits(pred_tensor, target_tensor)

    def compute_losses(self, raw_preds: HeadOutput, targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        roi_boxes = raw_preds.extra["roi_boxes"]
        logits = raw_preds.logits
        box_deltas = raw_preds.values
        if logits is None or box_deltas is None:
            raise ValueError("RCNNCriterion requires logits and values for Fast/Faster/Mask R-CNN losses.")

        sampled_global_indices: list[torch.Tensor] = []
        cls_targets_list: list[torch.Tensor] = []
        matched_gt_global = torch.zeros((roi_boxes.shape[0],), dtype=torch.long, device=roi_boxes.device)

        for image_index, target in enumerate(targets):
            image_idx = torch.where(roi_boxes[:, 0].long() == image_index)[0]
            if image_idx.numel() == 0:
                continue
            proposals = roi_boxes[image_idx, 1:]

            labels, matched_gt = self.roi_matcher.match(
                proposals,
                target["boxes"],
                target["labels"].clamp(min=0, max=self.cfg.num_classes - 1),
                background_label=self.cfg.num_classes,
            )
            sampled = self.roi_sampler.sample(
                _labels_to_binary(labels, num_classes=self.cfg.num_classes),
                positive_value=1,
                negative_value=0,
            )
            sampled_global = image_idx[sampled]
            sampled_global_indices.append(sampled_global)
            cls_targets_list.append(labels[sampled])
            matched_gt_global[image_idx] = matched_gt

        if sampled_global_indices:
            sampled_global = torch.cat(sampled_global_indices, dim=0)
            cls_targets_all = torch.cat(cls_targets_list, dim=0)
        else:
            sampled_global = torch.zeros((0,), dtype=torch.long, device=roi_boxes.device)
            cls_targets_all = torch.zeros((0,), dtype=torch.long, device=roi_boxes.device)

        if sampled_global.numel() == 0:
            roi_cls_loss = logits.sum() * 0.0
            roi_box_loss = box_deltas.sum() * 0.0
        else:
            roi_cls_loss = F.cross_entropy(logits[sampled_global], cls_targets_all)

            gt_boxes_for_samples = []
            for idx, global_idx in enumerate(sampled_global.tolist()):
                batch_idx = int(roi_boxes[global_idx, 0].item())
                if cls_targets_all[idx] >= self.cfg.num_classes:
                    gt_boxes_for_samples.append(roi_boxes[global_idx, 1:].detach())
                else:
                    gt_idx = int(matched_gt_global[global_idx].item())
                    gt_boxes_for_samples.append(targets[batch_idx]["boxes"][gt_idx])
            gt_boxes_tensor = torch.stack(gt_boxes_for_samples, dim=0)
            roi_box_loss = self._box_reg_loss(
                pred_deltas=box_deltas[sampled_global],
                proposals=roi_boxes[sampled_global, 1:],
                gt_boxes=gt_boxes_tensor,
                cls_targets=cls_targets_all,
            )

        losses: dict[str, torch.Tensor] = {
            "roi_cls_loss": roi_cls_loss,
            "roi_box_loss": roi_box_loss,
        }

        rpn_output = raw_preds.extra.get("rpn")
        if isinstance(rpn_output, FeatureMaps):
            losses.update(self._rpn_losses(rpn_output, targets))

        if raw_preds.masks is not None:
            losses["roi_mask_loss"] = self.cfg.mask_loss_weight * self._mask_loss(
                pred_masks=raw_preds.masks,
                sampled_global_indices=sampled_global,
                cls_targets_all=cls_targets_all,
                matched_gt_global=matched_gt_global,
                targets=targets,
                roi_boxes=roi_boxes,
            )

        return losses

    def compute_metrics(self, raw_preds: HeadOutput, targets: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        if raw_preds.logits is None:
            return {}
        with torch.no_grad():
            pred = raw_preds.logits.argmax(dim=1)
            valid = pred < self.cfg.num_classes
            fg_ratio = valid.float().mean().item() if pred.numel() > 0 else 0.0
        return {"pred_fg_ratio": fg_ratio}


rcnn_criterion_configs = {
    "rcnn_criterion": RCNNCriterionConfig(),
}


@register_criterion(config=rcnn_criterion_configs["rcnn_criterion"])
def rcnn_criterion(cfg: RCNNCriterionConfig) -> RCNNCriterion:
    return RCNNCriterion(cfg)
