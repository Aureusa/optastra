from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.component_ref import ComponentRef, component_field, ComponentRefConfigMixin
from ..backbones.base import Backbone
from ..heads.base import Head
from ..necks.base import Neck
from ..nn.features import HeadOutput
from ..region_extractors.base import RegionExtractor
from ..detection import keys
from .base import Architecture, add_gt_boxes_to_rois, resolve_image_sizes
from .faster_rcnn import FPN_ROI_STAGES


__all__ = ["FastRCNN", "FastRCNNConfig", "fast_rcnn_r18_fpn", "fast_rcnn_r50_fpn"]


@dataclass
class FastRCNNConfig(ComponentRefConfigMixin):
    backbone: ComponentRef = component_field(Backbone, default_name="resnet50")
    neck: ComponentRef | None = component_field(Neck, default_name="fpn")
    region_extractor: ComponentRef = component_field(RegionExtractor, default_name="roi_align")
    roi_box_head: ComponentRef = component_field(Head, default_name="roi_box_head")
    num_classes: int = 80  # foreground classes; must match the task's num_classes


class FastRCNN(Architecture):
    """Detectron-style Fast R-CNN: no proposal generator, expects RoIs as input."""

    def __init__(self, cfg: FastRCNNConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = cfg.backbone.resolve(Backbone)
        if cfg.neck is not None:
            self.neck = cfg.neck.resolve(Neck, in_spec=self.backbone.out_spec)
            detector_in_spec = self.neck.out_spec
        else:
            self.neck = None
            detector_in_spec = self.backbone.out_spec

        self.region_extractor = cfg.region_extractor.resolve(RegionExtractor, in_spec=detector_in_spec)
        self.roi_head = cfg.roi_box_head.resolve(Head, in_spec=self.region_extractor.out_spec, num_classes=cfg.num_classes)

    def info(self) -> str:
        info_str = f"FastRCNN Architecture:\n"
        info_str += f"(Backbone) {self.backbone.info()}\n"
        if self.neck is not None:
            info_str += f"(Neck) {self.neck.info()}\n"
        info_str += f"(Region Extractor) {self.region_extractor.info()}\n"
        info_str += f"(ROI Head) {self.roi_head.info()}\n"
        return info_str

    def forward(
        self,
        images: torch.Tensor,
        rois: torch.Tensor | None = None,
        image_sizes: list[tuple[int, int]] | torch.Tensor | None = None,
        gt_boxes: list[torch.Tensor] | None = None,
    ) -> HeadOutput:
        """
        :param images: (N, C, H, W) batch, possibly padded.
        :param rois: (R, 5) precomputed proposals as (batch_index, x1, y1, x2, y2);
                     the ragged collate builds them from each sample's ``target["proposals"]``.
        :param image_sizes: per-image (h, w) before padding.
        :param gt_boxes: per-image (G_i, 4) GT boxes appended to the ROIs (training only).
        """
        if rois is None:
            raise ValueError(
                "FastRCNN requires explicit rois input. Put per-image (P, 4) proposals under "
                "Sample.target['proposals'] and the ragged collate will batch them as 'rois'."
            )
        image_sizes = resolve_image_sizes(images, image_sizes)
        if gt_boxes is not None:
            rois = add_gt_boxes_to_rois(rois.float(), gt_boxes)
        features = self.backbone(images)
        detector_features = self.neck(features) if self.neck is not None else features
        roi_features = self.region_extractor(detector_features, rois)
        roi_output = self.roi_head(roi_features)
        extra = {keys.ROI_BOXES: rois, keys.IMAGE_SIZES: image_sizes}
        return HeadOutput(logits=roi_output.logits, values=roi_output.values, extra=extra)


fast_rcnn_configs = {
    "fast_rcnn_r18_fpn": FastRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stages": FPN_ROI_STAGES, "output_size": 7}),
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
        num_classes=5,
    ),
    "fast_rcnn_r50_fpn": FastRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stages": FPN_ROI_STAGES, "output_size": 7}),
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
        num_classes=5,
    ),
}


@Architecture.register(config=fast_rcnn_configs["fast_rcnn_r18_fpn"])
def fast_rcnn_r18_fpn(cfg: FastRCNNConfig) -> FastRCNN:
    return FastRCNN(cfg)


@Architecture.register(config=fast_rcnn_configs["fast_rcnn_r50_fpn"])
def fast_rcnn_r50_fpn(cfg: FastRCNNConfig) -> FastRCNN:
    return FastRCNN(cfg)
