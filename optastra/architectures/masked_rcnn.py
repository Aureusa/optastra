from dataclasses import dataclass, field
from typing import Any
import torch

from ..proposal_generators.base import ProposalGenerator

from ..core.component_ref import ComponentRef, resolve_component, component_field
from ..backbones import Backbone
from ..necks.base import Neck
from ..heads.base import Head
from ..nn.features import HeadOutput
from ._registry import register_architecture
from .faster_rcnn import FasterRCNN, FasterRCNNConfig
from ..region_extractors.base import RegionExtractor


@dataclass
class MaskRCNNConfig(FasterRCNNConfig):
    mask_head: ComponentRef = component_field(Head, default_name="mask_rcnn_head")
    mask_region_extractor: ComponentRef = component_field(
        RegionExtractor,
        default_name="roi_align",
        default_overrides={"output_size": 14} # default to 14x14 since the default box head output is 7x7
    )


class MaskRCNN(FasterRCNN):
    def __init__(self, cfg: MaskRCNNConfig):
        super().__init__(cfg)
        self.cfg = cfg
        detector_in_spec = self.neck.out_spec if self.neck is not None else self.backbone.out_spec

        self.mask_region_extractor = resolve_component(cfg, "mask_region_extractor", in_spec=detector_in_spec)
        self.mask_head = resolve_component(cfg, "mask_head", in_spec=self.mask_region_extractor.out_spec, num_classes=cfg.num_classes)

    def info(self) -> str:
        info_str = f"MaskRCNN Architecture:\n"
        info_str += f"(Backbone) {self.backbone.info()}\n"
        if self.neck is not None:
            info_str += f"(Neck) {self.neck.info()}\n"
        info_str += f"(Proposal Generator) {self.proposal_generator.info()}\n"
        info_str += f"(Region Extractor) {self.region_extractor.info()}\n"
        info_str += f"(ROI Head) {self.roi_head.info()}\n"
        info_str += f"(Mask Region Extractor) {self.mask_region_extractor.info()}\n"
        info_str += f"(Mask Head) {self.mask_head.info()}\n"
        return info_str

    def forward(self, images: torch.Tensor, rois: torch.Tensor | None = None) -> HeadOutput:
        detector_features, rpn_outputs = self._forward_detector(images)
        roi_boxes, box_features = self._forward_roi_features(detector_features, rpn_outputs, rois)
        mask_features = self.mask_region_extractor(detector_features, roi_boxes)

        box_output = self.roi_head(box_features)
        mask_output = self.mask_head(mask_features)

        extra = {
            "rpn": rpn_outputs,
            "roi_boxes": roi_boxes,
        }
        return HeadOutput(logits=box_output.logits, values=box_output.values, masks=mask_output.masks, extra=extra)


mask_rcnn_configs = {
    "mask_rcnn_r18_fpn": MaskRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
    ),
    "mask_rcnn_r50_fpn": MaskRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
    ),
    "mask_rcnn_r18_c5": MaskRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=None,
        proposal_generator=ComponentRef("rpn", {"in_features": ("C5",)}),
        region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 7}),
    ),
    "mask_rcnn_r50_c5": MaskRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=None,
        proposal_generator=ComponentRef("rpn", {"in_features": ("C5",)}),
        region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 7}),
    ),
}


@register_architecture(config=mask_rcnn_configs["mask_rcnn_r18_fpn"])
def mask_rcnn_r18_fpn(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)


@register_architecture(config=mask_rcnn_configs["mask_rcnn_r50_fpn"])
def mask_rcnn_r50_fpn(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)


@register_architecture(config=mask_rcnn_configs["mask_rcnn_r18_c5"])
def mask_rcnn_r18_c5(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)


@register_architecture(config=mask_rcnn_configs["mask_rcnn_r50_c5"])
def mask_rcnn_r50_c5(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)
