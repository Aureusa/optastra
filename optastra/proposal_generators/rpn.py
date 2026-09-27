import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence

from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.blocks.geometry.boxes import (
    apply_deltas_to_anchors,
    batched_nms,
    clip_boxes_to_image,
    flatten_anchor_predictions,
    generate_anchors,
    remove_small_boxes,
)
from ..detection import keys
from ..nn.features import FeatureMaps, FeatureSpec
from .base import ProposalGenerator


__all__ = ["RPN", "RPNConfig", "rpn"]


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

    Proposals are selected per image (top-k per level, NMS across levels) and
    are detached: gradients reach the RPN only through its own losses.
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
        if box_dim != 4:
            raise ValueError(f"RPN only supports axis-aligned XYXY boxes (box_dim=4), got box_dim={box_dim}.")
        conv_dims = cfg.conv_dims

        self.in_features = stage_names
        self.in_strides = {name: in_spec.strides[name] for name in self.in_features}
        self.cfg = cfg
        self.num_anchors = num_anchors
        self.box_dim = box_dim
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
        scores: torch.Tensor,
        deltas: torch.Tensor,
        anchors: torch.Tensor,
        num_anchors_per_level: list[int],
        image_size: tuple[int, int],
        image_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the proposals of one image.

        :param scores: (A,) objectness logits of this image, all levels concatenated.
        :param deltas: (A, box_dim) box deltas, same order as ``scores``.
        :param anchors: (A, 4) anchors, same order as ``scores``.
        :param num_anchors_per_level: split sizes of the concatenated anchors.
        :param image_size: (h, w) of this image before batch padding; boxes are clipped to it.
        :param image_index: batch index written into column 0 of the proposals.
        :return: proposals (P, 5) as (batch_index, x1, y1, x2, y2) and their scores (P,).
        """
        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_level_ids: list[torch.Tensor] = []

        per_level = zip(
            scores.split(num_anchors_per_level),
            deltas.split(num_anchors_per_level),
            anchors.split(num_anchors_per_level),
        )
        for level_id, (level_scores, level_deltas, level_anchors) in enumerate(per_level):
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
            level_scores_kept = top_scores[keep]
            level_tensor = torch.full((level_scores_kept.numel(),), level_id, dtype=torch.int64, device=boxes.device)
            all_boxes.append(boxes)
            all_scores.append(level_scores_kept)
            all_level_ids.append(level_tensor)

        if not all_boxes:
            return scores.new_zeros((0, 5)), scores.new_zeros((0,))

        boxes = torch.cat(all_boxes, dim=0)
        kept_scores = torch.cat(all_scores, dim=0)
        level_ids = torch.cat(all_level_ids, dim=0)

        keep = batched_nms(boxes, kept_scores, level_ids, iou_threshold=self.cfg.nms_thresh)
        keep = keep[: self.cfg.post_nms_topk]
        boxes = boxes[keep]
        kept_scores = kept_scores[keep]

        batch_index = torch.full((boxes.shape[0], 1), image_index, dtype=boxes.dtype, device=boxes.device)
        proposals = torch.cat((batch_index, boxes), dim=1)
        return proposals, kept_scores

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        """Predict objectness + deltas for every anchor and, if the per-image sizes
        are known (``features.extra[IMAGE_SIZES]``), select proposals.

        The raw per-level maps stay in ``feature_maps`` (``{level}_objectness``,
        ``{level}_deltas``); the flattened, anchor-aligned tensors the loss and the
        proposal selection use go into ``extra`` -- see :mod:`optastra.detection.keys`.
        """
        objectness_maps: dict[str, torch.Tensor] = {}
        delta_maps: dict[str, torch.Tensor] = {}
        flat_objectness: list[torch.Tensor] = []
        flat_deltas: list[torch.Tensor] = []
        level_anchors: list[torch.Tensor] = []

        for name in self.in_features:
            feat = features.feature_maps[name]
            hidden = self.conv(feat)
            objectness = self.cls_logits(hidden)
            deltas = self.bbox_pred(hidden)
            objectness_maps[f"{name}_objectness"] = objectness
            delta_maps[f"{name}_deltas"] = deltas

            # Reorder to generate_anchors() order so row i of every tensor is anchor i.
            flat_objectness.append(flatten_anchor_predictions(objectness, 1).squeeze(-1))
            flat_deltas.append(flatten_anchor_predictions(deltas, self.box_dim))
            level_anchors.append(self._generate_level_anchors(name, feat))

        objectness_logits = torch.cat(flat_objectness, dim=1)  # (N, A)
        bbox_deltas = torch.cat(flat_deltas, dim=1)            # (N, A, box_dim)
        anchors = torch.cat(level_anchors, dim=0)              # (A, 4)
        num_anchors_per_level = [a.shape[0] for a in level_anchors]

        extra: dict[str, object] = {
            keys.ANCHORS: anchors,
            keys.OBJECTNESS_LOGITS: objectness_logits,
            keys.BBOX_DELTAS: bbox_deltas,
            keys.NUM_ANCHORS_PER_LEVEL: num_anchors_per_level,
            keys.BOX_CODER_WEIGHTS: tuple(self.cfg.bbox_reg_weights),
        }

        image_sizes = features.extra.get(keys.IMAGE_SIZES) if features.extra else None
        if image_sizes is not None:
            if len(image_sizes) != objectness_logits.shape[0]:
                raise ValueError(
                    f"Got {len(image_sizes)} image sizes for a batch of {objectness_logits.shape[0]} images."
                )
            extra[keys.IMAGE_SIZES] = image_sizes

            # Proposals are *inputs* to the ROI stage, not a differentiable function
            # of the RPN: detach (and select in fp32) so the ROI losses never
            # backpropagate into the RPN through the box coordinates.
            scores = objectness_logits.detach().float()
            deltas = bbox_deltas.detach().float()
            proposals_per_image: list[torch.Tensor] = []
            scores_per_image: list[torch.Tensor] = []
            for image_index, image_size in enumerate(image_sizes):
                image_props, image_scores = self._propose_for_image(
                    scores=scores[image_index],
                    deltas=deltas[image_index],
                    anchors=anchors,
                    num_anchors_per_level=num_anchors_per_level,
                    image_size=(int(image_size[0]), int(image_size[1])),
                    image_index=image_index,
                )
                proposals_per_image.append(image_props)
                scores_per_image.append(image_scores)

            extra[keys.PROPOSALS] = torch.cat(proposals_per_image, dim=0)
            extra[keys.PROPOSAL_SCORES] = torch.cat(scores_per_image, dim=0)

        return FeatureMaps(feature_maps={**objectness_maps, **delta_maps}, extra=extra)


rpn_configs = {
    "rpn": RPNConfig(),
}


@ProposalGenerator.register(config=rpn_configs["rpn"])
def rpn(in_spec: FeatureSpec, cfg: RPNConfig) -> RPN:
    return RPN(in_spec, cfg)
    