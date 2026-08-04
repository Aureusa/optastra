from dataclasses import dataclass, field
from typing import Any

from ..heads.base import Head
from ..nn.features import HeadOutput
from ._registry import register_architecture
from .faster_rcnn import FasterRCNN, FasterRCNNConfig
from ..region_extractors.base import RegionExtractor


@dataclass
class MaskRCNNConfig(FasterRCNNConfig):
	mask_head_name: str = "mask_rcnn_head"
	mask_head_overrides: dict[str, Any] = field(default_factory=dict)
	mask_region_extractor_name: str = "roi_align"
	mask_region_extractor_overrides: dict[str, Any] = field(default_factory=dict)


class MaskRCNN(FasterRCNN):
	def __init__(self, cfg: MaskRCNNConfig):
		super().__init__(cfg)
		self.cfg = cfg
		detector_in_spec = self.neck.out_spec if self.neck is not None else self.backbone.out_spec

		mask_head_overrides = dict(cfg.mask_head_overrides)
		mask_head_overrides.setdefault("num_classes", cfg.num_classes)

		self.mask_region_extractor = RegionExtractor.create(
			cfg.mask_region_extractor_name,
			in_spec=detector_in_spec,
			**cfg.mask_region_extractor_overrides,
		)
	
		self.mask_head = Head.create(
			cfg.mask_head_name,
			in_spec=self.mask_region_extractor.out_spec,
			**mask_head_overrides,
		)

	def _forward_roi_features(self, detector_features, rpn_outputs, rois=None):
		box_rois = rpn_outputs.boxes if rois is None else rois
		box_features = self.region_extractor(detector_features, box_rois)
		mask_features = self.mask_region_extractor(detector_features, box_rois)
		return box_features, mask_features

	def forward(self, images, rois=None) -> HeadOutput:
		detector_features, rpn_outputs = self._forward_detector(images)
		box_features, mask_features = self._forward_roi_features(detector_features, rpn_outputs, rois)

		box_output = self.roi_head(box_features)
		mask_output = self.mask_head(mask_features)

		extra = {}
		if box_output.extra is not None:
			extra.update(box_output.extra)
		if mask_output.extra is not None:
			extra.update(mask_output.extra)

		output = HeadOutput(
			logits=box_output.logits,
			values=box_output.values,
			boxes=box_output.boxes,
			scores=box_output.scores,
			masks=mask_output.masks,
			embedding=box_output.embedding,
			extra=extra,
		)
		return self._attach_rpn_outputs(output, rpn_outputs)


mask_rcnn_configs = {
	"mask_rcnn_r18_fpn": MaskRCNNConfig(
		backbone_name="resnet18",
		neck_name="fpn",
		region_extractor_overrides={"stage": "P2", "output_size": 7},
	),
	"mask_rcnn_r50_fpn": MaskRCNNConfig(
		backbone_name="resnet50",
		neck_name="fpn",
		region_extractor_overrides={"stage": "P2", "output_size": 7},
	),
	"mask_rcnn_r18_c5": MaskRCNNConfig(
		backbone_name="resnet18",
		neck_name=None,
		proposal_generator_overrides={"in_features": ("C5",)},
		region_extractor_overrides={"stage": "C5", "output_size": 7},
	),
	"mask_rcnn_r50_c5": MaskRCNNConfig(
		backbone_name="resnet50",
		neck_name=None,
		proposal_generator_overrides={"in_features": ("C5",)},
		region_extractor_overrides={"stage": "C5", "output_size": 7},
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
