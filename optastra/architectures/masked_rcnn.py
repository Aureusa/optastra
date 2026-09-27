from dataclasses import dataclass
import torch

from ..core.component_ref import ComponentRef, component_field
from ..heads.base import Head
from ..nn.features import HeadOutput
from ..detection import keys
from .base import Architecture, resolve_image_sizes
from .faster_rcnn import FPN_ROI_STAGES, FasterRCNN, FasterRCNNConfig
from ..region_extractors.base import RegionExtractor


__all__ = ["MaskRCNN", "MaskRCNNConfig", "mask_rcnn_r18_fpn", "mask_rcnn_r50_fpn", "mask_rcnn_r18_c5", "mask_rcnn_r50_c5"]


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

        self.mask_region_extractor = cfg.mask_region_extractor.resolve(RegionExtractor, in_spec=detector_in_spec)
        self.mask_head = cfg.mask_head.resolve(Head, in_spec=self.mask_region_extractor.out_spec, num_classes=cfg.num_classes)

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

    def forward(
        self,
        images: torch.Tensor,
        rois: torch.Tensor | None = None,
        image_sizes: list[tuple[int, int]] | torch.Tensor | None = None,
        gt_boxes: list[torch.Tensor] | None = None,
    ) -> HeadOutput:
        """Same inputs as :meth:`FasterRCNN.forward`; additionally returns per-ROI mask logits."""
        image_sizes = resolve_image_sizes(images, image_sizes)
        detector_features, rpn_outputs = self._forward_detector(images, image_sizes)
        roi_boxes, box_features = self._forward_roi_features(detector_features, rpn_outputs, rois, gt_boxes)
        mask_features = self.mask_region_extractor(detector_features, roi_boxes)

        box_output = self.roi_head(box_features)
        mask_output = self.mask_head(mask_features)

        extra = {
            keys.RPN: rpn_outputs,
            keys.ROI_BOXES: roi_boxes,
            keys.IMAGE_SIZES: image_sizes,
        }
        return HeadOutput(logits=box_output.logits, values=box_output.values, masks=mask_output.masks, extra=extra)


# The mask extractor must pool from the same stage(s) as the box extractor --
# only the output resolution differs (14x14 -> 28x28 masks after the head's deconv).
mask_rcnn_configs = {
    "mask_rcnn_r18_fpn": MaskRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stages": FPN_ROI_STAGES, "output_size": 7}),
        mask_region_extractor=ComponentRef("roi_align", {"stages": FPN_ROI_STAGES, "output_size": 14}),
    ),
    "mask_rcnn_r50_fpn": MaskRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=ComponentRef("fpn"),
        region_extractor=ComponentRef("roi_align", {"stages": FPN_ROI_STAGES, "output_size": 7}),
        mask_region_extractor=ComponentRef("roi_align", {"stages": FPN_ROI_STAGES, "output_size": 14}),
    ),
    "mask_rcnn_r18_c5": MaskRCNNConfig(
        backbone=ComponentRef("resnet18"),
        neck=None,
        proposal_generator=ComponentRef("rpn", {"in_features": ("C5",)}),
        region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 7}),
        mask_region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 14}),
    ),
    "mask_rcnn_r50_c5": MaskRCNNConfig(
        backbone=ComponentRef("resnet50"),
        neck=None,
        proposal_generator=ComponentRef("rpn", {"in_features": ("C5",)}),
        region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 7}),
        mask_region_extractor=ComponentRef("roi_align", {"stage": "C5", "output_size": 14}),
    ),
}


@Architecture.register(config=mask_rcnn_configs["mask_rcnn_r18_fpn"])
def mask_rcnn_r18_fpn(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)


@Architecture.register(config=mask_rcnn_configs["mask_rcnn_r50_fpn"])
def mask_rcnn_r50_fpn(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)


@Architecture.register(config=mask_rcnn_configs["mask_rcnn_r18_c5"])
def mask_rcnn_r18_c5(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)


@Architecture.register(config=mask_rcnn_configs["mask_rcnn_r50_c5"])
def mask_rcnn_r50_c5(cfg: MaskRCNNConfig) -> MaskRCNN:
    return MaskRCNN(cfg)
