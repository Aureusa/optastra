from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ..core.component_ref import ComponentRef, resolve_component, component_field, ComponentRefConfigMixin
from ..backbones.base import Backbone
from ..heads.base import Head
from ..necks.base import Neck
from ..nn.features import HeadOutput
from ..region_extractors.base import RegionExtractor
from ._registry import register_architecture
from .base import Architecture


@dataclass
class FastRCNNConfig(ComponentRefConfigMixin):
    backbone: ComponentRef = component_field(Backbone, default_name="resnet50")
    neck: ComponentRef | None = component_field(Neck, default_name="fpn")
    region_extractor: ComponentRef = component_field(RegionExtractor, default_name="roi_align")
    roi_box_head: ComponentRef = component_field(Head, default_name="roi_box_head")
    num_classes: int = 91


class FastRCNN(Architecture):
    """Detectron-style Fast R-CNN: no proposal generator, expects RoIs as input."""

    def __init__(self, cfg: FastRCNNConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = resolve_component(cfg, "backbone")
        if cfg.neck is not None:
            self.neck = resolve_component(cfg, "neck", in_spec=self.backbone.out_spec)
            detector_in_spec = self.neck.out_spec
        else:
            self.neck = None
            detector_in_spec = self.backbone.out_spec

        self.region_extractor = resolve_component(cfg, "region_extractor", in_spec=detector_in_spec)
        self.roi_head = resolve_component(cfg, "roi_box_head", in_spec=self.region_extractor.out_spec, num_classes=cfg.num_classes)

    def info(self) -> str:
        info_str = f"FastRCNN Architecture:\n"
        info_str += f"(Backbone) {self.backbone.info()}\n"
        if self.neck is not None:
            info_str += f"(Neck) {self.neck.info()}\n"
        info_str += f"(Region Extractor) {self.region_extractor.info()}\n"
        info_str += f"(ROI Head) {self.roi_head.info()}\n"
        return info_str

    def forward(self, images: torch.Tensor, rois: torch.Tensor) -> HeadOutput:
        if rois is None:
            raise ValueError("FastRCNN requires explicit rois input.")
        features = self.backbone(images)
        detector_features = self.neck(features) if self.neck is not None else features
        roi_features = self.region_extractor(detector_features, rois)
        roi_output = self.roi_head(roi_features)
        return HeadOutput(logits=roi_output.logits, values=roi_output.values, extra={"roi_boxes": rois})


fast_rcnn_configs = {
    "fast_rcnn_r18_fpn": FastRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
        num_classes=5,
    ),
    "fast_rcnn_r50_fpn": FastRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
        num_classes=5,
    ),
}


@register_architecture(config=fast_rcnn_configs["fast_rcnn_r18_fpn"])
def fast_rcnn_r18_fpn(cfg: FastRCNNConfig) -> FastRCNN:
    return FastRCNN(cfg)


@register_architecture(config=fast_rcnn_configs["fast_rcnn_r50_fpn"])
def fast_rcnn_r50_fpn(cfg: FastRCNNConfig) -> FastRCNN:
    return FastRCNN(cfg)
