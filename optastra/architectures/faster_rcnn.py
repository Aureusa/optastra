import torch
from dataclasses import dataclass, field
from typing import Any

from ..core.component_ref import ComponentRef
from ..backbones.base import Backbone
from ..necks.base import Neck
from ..heads.base import Head
from ..nn.features import FeatureMaps, HeadOutput
from ..proposal_generators.base import ProposalGenerator
from ..region_extractors.base import RegionExtractor
from .base import Architecture
from ._registry import register_architecture


@dataclass
class FasterRCNNConfig:
    backbone: ComponentRef = field(default_factory=lambda: ComponentRef("resnet50"))
    neck: ComponentRef | None = field(default_factory=lambda: ComponentRef("fpn"))
    proposal_generator: ComponentRef = field(default_factory=lambda: ComponentRef("rpn"))
    region_extractor: ComponentRef = field(default_factory=lambda: ComponentRef("roi_align"))
    roi_box_head: ComponentRef = field(default_factory=lambda: ComponentRef("roi_box_head"))
    num_classes: int = 91


class FasterRCNN(Architecture):
    def __init__(
            self,
            cfg: FasterRCNNConfig,
        ):
        super().__init__()
        self.cfg = cfg

        self.backbone = cfg.backbone.resolve(Backbone)

        if cfg.neck is not None:
            self.neck = cfg.neck.resolve(Neck, in_spec=self.backbone.out_spec)
            detector_in_spec = self.neck.out_spec
        else:
            self.neck = None
            detector_in_spec = self.backbone.out_spec

        self.proposal_generator = cfg.proposal_generator.resolve(ProposalGenerator, in_spec=detector_in_spec)
        self.region_extractor = cfg.region_extractor.resolve(RegionExtractor, in_spec=detector_in_spec)
        self.roi_head = cfg.roi_box_head.resolve(Head, in_spec=self.region_extractor.out_spec, num_classes=cfg.num_classes)

    def info(self) -> str:
        info_str = f"FasterRCNN Architecture:\n"
        info_str += f"(Backbone) {self.backbone.info()}\n"
        if self.neck is not None:
            info_str += f"(Neck) {self.neck.info()}\n"
        info_str += f"(Proposal Generator) {self.proposal_generator.info()}\n"
        info_str += f"(Region Extractor) {self.region_extractor.info()}\n"
        info_str += f"(ROI Head) {self.roi_head.info()}\n"
        return info_str

    def _forward_detector(self, images: torch.Tensor):
        features = self.backbone(images)
        detector_features = self.neck(features) if self.neck is not None else features
        detector_features.extra["image_size"] = (int(images.shape[-2]), int(images.shape[-1]))
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
    def _resolve_rois(rois: torch.Tensor | None, rpn_outputs) -> torch.Tensor:
        if rois is not None:
            return rois

        if rpn_outputs.extra and "proposals" in rpn_outputs.extra and isinstance(rpn_outputs.extra["proposals"], torch.Tensor):
            return rpn_outputs.extra["proposals"]

        if "proposals" in rpn_outputs.feature_maps and isinstance(rpn_outputs.feature_maps["proposals"], torch.Tensor):
            return rpn_outputs.feature_maps["proposals"]

        raise ValueError(
            "FasterRCNN requires proposal boxes for ROI extraction. Provide 'rois' to forward(), "
            "or use a proposal generator that returns a tensor under `FeatureMaps.extra['proposals']`."
        )

    def forward(self, images: torch.Tensor, rois: torch.Tensor | None = None) -> HeadOutput:
        detector_features, rpn_outputs = self._forward_detector(images)
        roi_boxes, roi_features = self._forward_roi_features(detector_features, rpn_outputs, rois)
        roi_output = self.roi_head(roi_features)

        extra = {
            "rpn": rpn_outputs,
            "roi_boxes": roi_boxes,
        }
        return HeadOutput(logits=roi_output.logits, values=roi_output.values, extra=extra)


faster_rcnn_configs = {
    "faster_rcnn_r18_fpn": FasterRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
    ),
    "faster_rcnn_r50_fpn": FasterRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
    ),
    "faster_rcnn_r18_c5": FasterRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=None,
        proposal_generator=ComponentRef("rpn", {"in_features": ("C5",)}),
        region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 7}),
    ),
    "faster_rcnn_r50_c5": FasterRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=None,
        proposal_generator=ComponentRef("rpn", {"in_features": ("C5",)}),
        region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 7}),
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
    