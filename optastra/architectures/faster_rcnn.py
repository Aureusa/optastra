import torch
from dataclasses import dataclass, field
import torch.nn as nn
from typing import Any

from ..backbones.base import Backbone
from ..necks.base import Neck
from ..heads.base import Head
from ..nn.features import FeatureMaps, FeatureSpec, HeadOutput
from ..nn.blocks.readout.mlp import MLP
from ..proposal_generators.base import ProposalGenerator
from ..region_extractors.base import RegionExtractor
from .base import Architecture
from ._registry import register_architecture


@dataclass
class FasterRCNNConfig:
    backbone_name: str = "resnet50"
    neck_name: str | None = "fpn"
    proposal_generator_name: str = "rpn"
    region_extractor_name: str = "roi_align"
    head_name: str = "vanilla_classification_head"
    box_head_name: str = "vanilla_box_regression_head"
    num_classes: int = 91
    fc_hidden_features: int = 256
    fc_num_layers: int = 2
    backbone_overrides: dict[str, Any] = field(default_factory=dict)
    neck_overrides: dict[str, Any] = field(default_factory=dict)
    proposal_generator_overrides: dict[str, Any] = field(default_factory=dict)
    region_extractor_overrides: dict[str, Any] = field(default_factory=dict)
    head_overrides: dict[str, Any] = field(default_factory=dict)
    box_head_overrides: dict[str, Any] = field(default_factory=dict)


class ROIHead(nn.Module):

    def __init__(
            self,
            in_spec: FeatureSpec,
            cls_head: str,
            bbox_head: str,
            hidden_features=256,
            num_layers=2,
            cls_head_overrides: dict[str, Any] | None = None,
            box_head_overrides: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.fc = MLP(
            in_features=in_spec.embed_dim,
            hidden_features=hidden_features,
            out_features=hidden_features,
            num_layers=num_layers,
        )
        shared_spec = FeatureSpec(embed_dim=hidden_features)
        self.cls = Head.create(
            cls_head,
            in_spec=shared_spec,
            **(cls_head_overrides or {}),
        )
        self.bbox = Head.create(
            bbox_head,
            in_spec=shared_spec,
            **(box_head_overrides or {}),
        )

    def forward(self, features: FeatureMaps) -> HeadOutput:
        shared = self.fc(features.pooled)
        shared_features = FeatureMaps(pooled=shared)
        cls_output = self.cls(shared_features)
        bbox_output = self.bbox(shared_features)
        return HeadOutput(
            logits=cls_output.logits,
            values=bbox_output.values,
            boxes=bbox_output.boxes,
            scores=cls_output.scores,
            masks=cls_output.masks,
            embedding=cls_output.embedding,
            extra={**cls_output.extra, **bbox_output.extra},
        )


class FasterRCNN(Architecture):
    def __init__(
            self,
            cfg: FasterRCNNConfig,
        ):
        super().__init__()
        self.cfg = cfg

        backbone_overrides = dict(cfg.backbone_overrides)
        neck_overrides = dict(cfg.neck_overrides)
        proposal_generator_overrides = dict(cfg.proposal_generator_overrides)
        region_extractor_overrides = dict(cfg.region_extractor_overrides)
        head_overrides = dict(cfg.head_overrides)
        box_head_overrides = dict(cfg.box_head_overrides)
        head_overrides["num_classes"] = cfg.num_classes
        box_head_overrides["num_classes"] = cfg.num_classes

        self.backbone = Backbone.create(cfg.backbone_name, **backbone_overrides)

        if cfg.neck_name is not None:
            self.neck = Neck.create(cfg.neck_name, in_spec=self.backbone.out_spec, **neck_overrides)
            detector_in_spec = self.neck.out_spec
        else:
            self.neck = None
            detector_in_spec = self.backbone.out_spec

        self.proposal_generator = ProposalGenerator.create(
            cfg.proposal_generator_name,
            in_spec=detector_in_spec,
            **proposal_generator_overrides,
        )
        self.region_extractor = RegionExtractor.create(
            cfg.region_extractor_name,
            in_spec=detector_in_spec,
            **region_extractor_overrides,
        )
        self.roi_head = ROIHead(
            in_spec=self.region_extractor.out_spec,
            cls_head=cfg.head_name,
            bbox_head=cfg.box_head_name,
            hidden_features=cfg.fc_hidden_features,
            num_layers=cfg.fc_num_layers,
            cls_head_overrides=head_overrides,
            box_head_overrides=box_head_overrides,
        )

    def _forward_detector(self, images: torch.Tensor):
        features = self.backbone(images)
        detector_features = self.neck(features) if self.neck is not None else features
        rpn_outputs = self.proposal_generator(detector_features)
        return detector_features, rpn_outputs

    def _forward_roi_features(
        self,
        detector_features: FeatureMaps,
        rpn_outputs: FeatureMaps,
        rois: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, FeatureMaps]:
        roi_boxes = self._resolve_rois(rois, rpn_outputs)
        roi_features = self.region_extractor(detector_features, roi_boxes)
        return roi_boxes, roi_features

    @staticmethod
    def _attach_rpn_outputs(output: HeadOutput, rpn_outputs: FeatureMaps) -> HeadOutput:
        if output.extra is None:
            output.extra = {}
        output.extra["rpn"] = rpn_outputs.feature_maps
        return output

    @staticmethod
    def _resolve_rois(rois: torch.Tensor | None, rpn_outputs) -> torch.Tensor:
        if rois is not None:
            return rois

        if "proposals" in rpn_outputs.extra and isinstance(rpn_outputs.extra["proposals"], torch.Tensor):
            return rpn_outputs.extra["proposals"]

        if "proposals" in rpn_outputs.feature_maps and isinstance(rpn_outputs.feature_maps["proposals"], torch.Tensor):
            return rpn_outputs.feature_maps["proposals"]

        raise ValueError(
            "FasterRCNN requires proposal boxes for ROI extraction. Provide 'rois' to forward(), "
            "or use a proposal generator that returns a tensor under `FeatureMaps.extra['proposals']`."
        )

    def forward(self, images: torch.Tensor, rois: torch.Tensor | None = None) -> HeadOutput:
        detector_features, rpn_outputs = self._forward_detector(images)
        _, roi_features = self._forward_roi_features(detector_features, rpn_outputs, rois)
        output = self.roi_head(roi_features)
        return self._attach_rpn_outputs(output, rpn_outputs)


faster_rcnn_configs = {
    "faster_rcnn_r18_fpn": FasterRCNNConfig(
        backbone_name="resnet18",
        neck_name="fpn",
        region_extractor_overrides={"stage": "P2", "output_size": 7},
    ),
    "faster_rcnn_r50_fpn": FasterRCNNConfig(
        backbone_name="resnet50",
        neck_name="fpn",
        region_extractor_overrides={"stage": "P2", "output_size": 7},
    ),
    "faster_rcnn_r18_c5": FasterRCNNConfig(
        backbone_name="resnet18",
        neck_name=None,
        proposal_generator_overrides={"in_features": ("C5",)},
        region_extractor_overrides={"stage": "C5", "output_size": 7},
    ),
    "faster_rcnn_r50_c5": FasterRCNNConfig(
        backbone_name="resnet50",
        neck_name=None,
        proposal_generator_overrides={"in_features": ("C5",)},
        region_extractor_overrides={"stage": "C5", "output_size": 7},
    ),
}


@register_architecture(config=faster_rcnn_configs["faster_rcnn_r18_fpn"])
def faster_rcnn_r18_fpn(cfg: FasterRCNNConfig) -> FasterRCNN:
    return FasterRCNN(cfg)


@register_architecture(config=faster_rcnn_configs["faster_rcnn_r50_fpn"])
def faster_rcnn_r50_fpn(cfg: FasterRCNNConfig) -> FasterRCNN:
    return FasterRCNN(cfg)


@register_architecture(config=faster_rcnn_configs["faster_rcnn_r18_c5"])
def faster_rcnn_r18_c5(cfg: FasterRCNNConfig) -> FasterRCNN:
    return FasterRCNN(cfg)


@register_architecture(config=faster_rcnn_configs["faster_rcnn_r50_c5"])
def faster_rcnn_r50_c5(cfg: FasterRCNNConfig) -> FasterRCNN:
    return FasterRCNN(cfg)
    