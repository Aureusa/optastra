import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence

from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.blocks.geometry.boxes import (
    apply_deltas_to_anchors,
    batched_nms,
    clip_boxes_to_image,
    generate_anchors,
    remove_small_boxes,
)
from ..nn.features import FeatureMaps, FeatureSpec
from .base import ProposalGenerator
from ._registry import register_proposal_generator


@dataclass
class RPNConfig:
    num_anchors: int | None = None
    box_dim: int = 4
    conv_dims: Sequence[int] = (-1,)
    in_features: tuple[str, ...] = ()
    anchor_scales: tuple[float, ...] = (8.0,)
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    pre_nms_topk: int = 1000
    post_nms_topk: int = 300
    nms_thresh: float = 0.7
    min_box_size: float = 1.0
    bbox_reg_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


class RPN(ProposalGenerator):
    """
    Simplified Detectron2-style RPN head.

    The module consumes one feature map or a list of feature maps and predicts:
    - objectness logits per anchor location
    - box deltas per anchor location
    """

    def __init__(
        self,
        in_spec: FeatureSpec,
        cfg: RPNConfig,
    ):
        super().__init__()
        in_spec.require("channels", "strides")

        stage_names = tuple(sorted(in_spec.channels.keys())) if not cfg.in_features else cfg.in_features
        for name in stage_names:
            if name not in in_spec.channels:
                raise ValueError(f"Requested feature '{name}' is missing from in_spec.channels")

        in_channels_per_stage = [in_spec.channels[name] for name in stage_names]
        if len(set(in_channels_per_stage)) != 1:
            raise ValueError("RPN expects selected input feature maps to have the same channel count")

        in_channels = in_channels_per_stage[0]
        inferred_num_anchors = len(cfg.anchor_scales) * len(cfg.aspect_ratios)
        num_anchors = inferred_num_anchors if cfg.num_anchors is None else cfg.num_anchors
        if num_anchors != inferred_num_anchors:
            raise ValueError(
                "num_anchors must equal len(anchor_scales) * len(aspect_ratios). "
                f"Got num_anchors={num_anchors}, inferred={inferred_num_anchors}."
            )
        box_dim = cfg.box_dim
        conv_dims = cfg.conv_dims

        self.in_features = stage_names
        self.in_strides = {name: in_spec.strides[name] for name in self.in_features}
        self.cfg = cfg
        self.num_anchors = num_anchors
        cur_channels = in_channels

        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            if out_channels <= 0:
                raise ValueError(f"Conv output channels must be > 0, got {out_channels}")
            self.conv = self._make_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            convs: list[nn.Module] = []
            for conv_dim in conv_dims:
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(f"Conv output channels must be > 0, got {out_channels}")
                convs.append(self._make_conv(cur_channels, out_channels))
                cur_channels = out_channels
            self.conv = nn.Sequential(*convs)

        self.cls_logits = nn.Conv2d(cur_channels, self.num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(cur_channels, self.num_anchors * box_dim, kernel_size=1, stride=1)

        self.out_spec = FeatureSpec(
            channels={
                **{f"{name}_objectness": num_anchors for name in self.in_features},
                **{f"{name}_deltas": num_anchors * box_dim for name in self.in_features},
            },
            strides={
                **{f"{name}_objectness": in_spec.strides[name] for name in self.in_features},
                **{f"{name}_deltas": in_spec.strides[name] for name in self.in_features},
            },
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    @staticmethod
    def _make_conv(in_channels: int, out_channels: int) -> nn.Module:
        return ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=None,
            activation="relu",
        )

    def _generate_level_anchors(self, level_name: str, feature_map: torch.Tensor) -> torch.Tensor:
        _, _, h, w = feature_map.shape
        stride = self.in_strides[level_name]
        return generate_anchors(
            height=h,
            width=w,
            stride=stride,
            scales=self.cfg.anchor_scales,
            aspect_ratios=self.cfg.aspect_ratios,
            device=feature_map.device,
        )

    def _propose_for_image(
        self,
        objectness: dict[str, torch.Tensor],
        deltas: dict[str, torch.Tensor],
        anchors: dict[str, torch.Tensor],
        image_size: tuple[int, int],
        image_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_level_ids: list[torch.Tensor] = []

        for level_id, level_name in enumerate(self.in_features):
            level_scores = objectness[level_name][image_index].reshape(-1)
            level_deltas = deltas[level_name][image_index].view(self.num_anchors, 4, *objectness[level_name].shape[-2:])
            level_deltas = level_deltas.permute(2, 3, 0, 1).reshape(-1, 4)
            level_anchors = anchors[level_name]

            topk = min(self.cfg.pre_nms_topk, level_scores.numel())
            if topk <= 0:
                continue
            top_scores, top_idx = torch.topk(level_scores, k=topk, dim=0)
            top_deltas = level_deltas[top_idx]
            top_anchors = level_anchors[top_idx]

            boxes = apply_deltas_to_anchors(top_deltas, top_anchors, weights=self.cfg.bbox_reg_weights)
            boxes = clip_boxes_to_image(boxes, image_size)
            keep = remove_small_boxes(boxes, self.cfg.min_box_size)
            if keep.numel() == 0:
                continue

            boxes = boxes[keep]
            scores = top_scores[keep]
            level_tensor = torch.full((scores.numel(),), level_id, dtype=torch.int64, device=scores.device)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_level_ids.append(level_tensor)

        if not all_boxes:
            empty_boxes = torch.zeros((0, 5), device=next(iter(objectness.values())).device)
            empty_scores = torch.zeros((0,), device=empty_boxes.device)
            return empty_boxes, empty_scores

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        level_ids = torch.cat(all_level_ids, dim=0)

        keep = batched_nms(boxes, scores, level_ids, iou_threshold=self.cfg.nms_thresh)
        keep = keep[: self.cfg.post_nms_topk]
        boxes = boxes[keep]
        scores = scores[keep]

        batch_index = torch.full((boxes.shape[0], 1), image_index, dtype=boxes.dtype, device=boxes.device)
        proposals = torch.cat((batch_index, boxes), dim=1)
        return proposals, scores

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        objectness_maps: dict[str, torch.Tensor] = {}
        delta_maps: dict[str, torch.Tensor] = {}
        anchor_maps: dict[str, torch.Tensor] = {}

        for name in self.in_features:
            feat = features.feature_maps[name]
            hidden = self.conv(feat)
            objectness_maps[f"{name}_objectness"] = self.cls_logits(hidden)
            delta_maps[f"{name}_deltas"] = self.bbox_pred(hidden)
            anchor_maps[name] = self._generate_level_anchors(name, feat)

        objectness_by_level = {n: objectness_maps[f"{n}_objectness"] for n in self.in_features}
        deltas_by_level = {n: delta_maps[f"{n}_deltas"] for n in self.in_features}

        extra: dict[str, object] = {
            "anchors": anchor_maps,
            "objectness_logits": objectness_by_level,
            "bbox_deltas": deltas_by_level,
        }

        image_size = features.extra.get("image_size") if features.extra else None
        if image_size is not None:
            extra["image_size"] = image_size
        if image_size is not None:
            batch_size = next(iter(objectness_by_level.values())).shape[0]
            proposals_per_image: list[torch.Tensor] = []
            scores_per_image: list[torch.Tensor] = []
            for image_index in range(batch_size):
                image_props, image_scores = self._propose_for_image(
                    objectness=objectness_by_level,
                    deltas=deltas_by_level,
                    anchors=anchor_maps,
                    image_size=image_size,
                    image_index=image_index,
                )
                proposals_per_image.append(image_props)
                scores_per_image.append(image_scores)

            extra["proposals"] = torch.cat(proposals_per_image, dim=0) if proposals_per_image else torch.zeros((0, 5))
            extra["proposal_scores"] = torch.cat(scores_per_image, dim=0) if scores_per_image else torch.zeros((0,))

        return FeatureMaps(feature_maps={**objectness_maps, **delta_maps}, extra=extra)


rpn_configs = {
    "rpn": RPNConfig(),
}


@register_proposal_generator(config=rpn_configs["rpn"])
def rpn(in_spec: FeatureSpec, cfg: RPNConfig) -> RPN:
    return RPN(in_spec, cfg)
    