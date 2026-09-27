import math

import torch
import torch.nn as nn
from dataclasses import dataclass
from torchvision.ops import roi_align as tv_roi_align

from ..nn.features import FeatureMaps, FeatureSpec
from .base import RegionExtractor


__all__ = ["ROIAlign", "ROIAlignConfig", "assign_boxes_to_levels"]


@dataclass
class ROIAlignConfig:
    output_size: int = 7
    # Single-level: pool every ROI from this stage ("" = first stage in in_spec).
    stage: str = ""
    # Multi-level (FPN): pool each ROI from one of these stages, chosen by box size.
    stages: tuple[str, ...] = ()
    spatial_scale: float | None = None  # single-level only; defaults to 1 / stride
    sampling_ratio: int = -1
    aligned: bool = True
    # FPN level assignment (Lin et al. 2017, eq. 1): a canonical_box_size box maps to canonical_level.
    canonical_box_size: float = 224.0
    canonical_level: int = 4


def assign_boxes_to_levels(
    boxes: torch.Tensor,
    min_level: int,
    max_level: int,
    canonical_box_size: float = 224.0,
    canonical_level: int = 4,
) -> torch.Tensor:
    """Pick the FPN level each box is pooled from (FPN paper, eq. 1).

        level = floor(canonical_level + log2(sqrt(box_area) / canonical_box_size))

    clamped to ``[min_level, max_level]``. With the defaults a 224x224 box goes
    to level 4 (P4, stride 16), a 112x112 box to P3, a 448x448 box to P5.

    :param boxes: (R, 4) XYXY boxes in image coordinates.
    :return: (R,) int64 index into the levels, i.e. ``level - min_level``.
    """
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    box_sizes = torch.sqrt(widths * heights)
    levels = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8))
    levels = levels.clamp(min=min_level, max=max_level)
    return levels.to(torch.int64) - min_level


class ROIAlign(RegionExtractor):
    """Region of Interest (RoI) Align layer for extracting fixed-size feature maps
    from variable-sized regions of interest (RoIs) in the input feature map.

    With ``cfg.stages`` naming several pyramid levels, every ROI is pooled from
    the single level that matches its size (see :func:`assign_boxes_to_levels`);
    otherwise all ROIs are pooled from ``cfg.stage``.
    """

    def __init__(self, in_spec: FeatureSpec, cfg: ROIAlignConfig):
        super().__init__()
        in_spec.require("channels", "strides")
        if cfg.stage and cfg.stages:
            raise ValueError("Set either 'stage' (single-level) or 'stages' (multi-level), not both.")

        stages = tuple(cfg.stages) or (cfg.stage or sorted(in_spec.channels.keys())[0],)
        for stage in stages:
            if stage not in in_spec.channels:
                raise ValueError(f"Requested feature '{stage}' is missing from in_spec.channels")
        channels = {in_spec.channels[stage] for stage in stages}
        if len(channels) != 1:
            raise ValueError(f"Multi-level ROIAlign needs equal channel counts, got {dict((s, in_spec.channels[s]) for s in stages)}")

        # Order levels fine -> coarse and check they form a contiguous pyramid (stride doubles per level).
        stages = tuple(sorted(stages, key=lambda s: in_spec.strides[s]))
        levels = [int(round(math.log2(in_spec.strides[s]))) for s in stages]
        if len(stages) > 1:
            if cfg.spatial_scale is not None:
                raise ValueError("'spatial_scale' is only supported for single-level ROIAlign.")
            if levels != list(range(levels[0], levels[0] + len(levels))):
                raise ValueError(f"Stages {stages} must have strides that double from level to level, got levels {levels}.")

        self.stages = stages
        self.stage = stages[0]  # kept for single-level callers / backwards compatibility
        self.min_level, self.max_level = levels[0], levels[-1]
        self.canonical_box_size = cfg.canonical_box_size
        self.canonical_level = cfg.canonical_level
        self.output_size = cfg.output_size
        self.spatial_scales = [
            cfg.spatial_scale if (cfg.spatial_scale is not None) else 1.0 / in_spec.strides[s] for s in stages
        ]
        self.spatial_scale = self.spatial_scales[0]
        self.sampling_ratio = cfg.sampling_ratio
        self.aligned = cfg.aligned

        num_channels = in_spec.channels[self.stage]
        self.out_spec = FeatureSpec(
            channels={"roi": num_channels},
            strides={"roi": 1},
            embed_dim=num_channels,
            num_tokens=cfg.output_size * cfg.output_size,
        )

    def _pool(self, feature_map: torch.Tensor, rois: torch.Tensor, spatial_scale: float) -> torch.Tensor:
        # Pool in fp32: ROI coordinates must never be rounded to a low-precision
        # feature dtype (bf16 has 8 mantissa bits -> ~1px error at x=256).
        return tv_roi_align(
            input=feature_map.float(),
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        ).to(feature_map.dtype)

    def forward(self, features: FeatureMaps, rois: torch.Tensor) -> FeatureMaps:
        """
        Forward pass of the ROIAlign layer.

        :param features: Input feature maps (FeatureMaps). Uses the configured stage(s).
        :param rois: Regions of interest of shape (num_rois, 5), where each ROI is
                     represented as (batch_index, x1, y1, x2, y2).
        :return: FeatureMaps with ``feature_maps["roi"]`` of shape
                 (num_rois, C, output_size, output_size) and ``pooled`` of shape (num_rois, C).
        """
        for stage in self.stages:
            if stage not in features.feature_maps:
                raise ValueError(f"FeatureMaps does not contain required stage '{stage}'")
            if features.feature_maps[stage].ndim != 4:
                raise ValueError(
                    f"'{stage}' feature must have shape (N, C, H, W), got {tuple(features.feature_maps[stage].shape)}"
                )

        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError(f"'rois' must have shape (num_rois, 5), got {tuple(rois.shape)}")

        first_map = features.feature_maps[self.stages[0]]
        rois = rois.to(device=first_map.device, dtype=torch.float32)

        if len(self.stages) == 1:
            roi_feats = self._pool(first_map, rois, self.spatial_scales[0])
        else:
            level_ids = assign_boxes_to_levels(
                rois[:, 1:],
                self.min_level,
                self.max_level,
                canonical_box_size=self.canonical_box_size,
                canonical_level=self.canonical_level,
            )
            roi_feats = first_map.new_zeros((rois.shape[0], first_map.shape[1], self.output_size, self.output_size))
            for level_id, (stage, scale) in enumerate(zip(self.stages, self.spatial_scales)):
                idx = torch.where(level_ids == level_id)[0]
                if idx.numel() == 0:
                    continue
                roi_feats[idx] = self._pool(features.feature_maps[stage], rois[idx], scale)

        pooled = roi_feats.mean(dim=(2, 3))  # (num_rois, C)
        return FeatureMaps(feature_maps={"roi": roi_feats}, pooled=pooled)


roi_align_configs = {
    "roi_align": ROIAlignConfig(),
}


@RegionExtractor.register(config=roi_align_configs["roi_align"])
def roi_align(in_spec: FeatureSpec, cfg: ROIAlignConfig) -> ROIAlign:
    return ROIAlign(in_spec, cfg)
