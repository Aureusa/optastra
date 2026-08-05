import torch
import torch.nn as nn
from dataclasses import dataclass
from torchvision.ops import roi_align as tv_roi_align

from ..nn.features import FeatureMaps, FeatureSpec
from .base import RegionExtractor
from ._registry import register_region_extractor


@dataclass
class ROIAlignConfig:
    output_size: int = 7
    stage: str = ""
    spatial_scale: float | None = None
    sampling_ratio: int = -1
    aligned: bool = True


class ROIAlign(RegionExtractor):
    """Region of Interest (RoI) Align layer for extracting fixed-size feature maps
    from variable-sized regions of interest (RoIs) in the input feature map."""

    def __init__(self, in_spec: FeatureSpec, cfg: ROIAlignConfig):
        super().__init__()
        in_spec.require("channels", "strides")

        stage = cfg.stage or sorted(in_spec.channels.keys())[0]
        if stage not in in_spec.channels:
            raise ValueError(f"Requested feature '{stage}' is missing from in_spec.channels")

        self.stage = stage
        self.output_size = cfg.output_size
        self.spatial_scale = cfg.spatial_scale if cfg.spatial_scale is not None else 1.0 / in_spec.strides[stage]
        self.sampling_ratio = cfg.sampling_ratio
        self.aligned = cfg.aligned

        self.out_spec = FeatureSpec(
            channels={"roi": in_spec.channels[stage]},
            strides={"roi": 1},
            embed_dim=in_spec.channels[stage],
            num_tokens=cfg.output_size * cfg.output_size,
        )

    def forward(self, features: FeatureMaps, rois: torch.Tensor) -> FeatureMaps:
        """
        Forward pass of the ROIAlign layer.
        
        :param features: Input feature maps (FeatureMaps). Uses the configured stage.
        :param rois: Regions of interest of shape (num_rois, 5), where each ROI is
                     represented as (batch_index, x1, y1, x2, y2).
        :return: Output feature map of shape (num_rois, C, output_size, output_size).
        """
        if self.stage not in features.feature_maps:
            raise ValueError(f"FeatureMaps does not contain required stage '{self.stage}'")

        feature_map = features.feature_maps[self.stage]
        if feature_map.ndim != 4:
            raise ValueError(f"'{self.stage}' feature must have shape (N, C, H, W), got {tuple(feature_map.shape)}")

        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError(f"'rois' must have shape (num_rois, 5), got {tuple(rois.shape)}")

        # torchvision ROIAlign expects floating-point ROI coordinates.
        rois = rois.to(device=feature_map.device, dtype=feature_map.dtype)

        roi_feats = tv_roi_align(
            input=feature_map,
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )

        pooled = roi_feats.mean(dim=(2, 3)) # (num_rois, C)
        return FeatureMaps(feature_maps={"roi": roi_feats}, pooled=pooled) # (num_rois, C, output_size, output_size)


roi_align_configs = {
    "roi_align": ROIAlignConfig(),
}


@register_region_extractor(config=roi_align_configs["roi_align"])
def roi_align(in_spec: FeatureSpec, cfg: ROIAlignConfig) -> ROIAlign:
    return ROIAlign(in_spec, cfg)
        