import importlib

import pytest
import torch

from optastra.nn.features import FeatureMaps, FeatureSpec
from optastra.region_extractors import RegionExtractor
from optastra.region_extractors.roi_align import ROIAlign, ROIAlignConfig, assign_boxes_to_levels

# The package re-exports the `roi_align` factory function under the module's name.
roi_align_module = importlib.import_module("optastra.region_extractors.roi_align")


def test_roi_align_returns_expected_shape():
    in_spec = FeatureSpec(channels={"P3": 3}, strides={"P3": 1})
    layer = ROIAlign(in_spec=in_spec, cfg=ROIAlignConfig(output_size=7, stage="P3", spatial_scale=1.0))
    features = FeatureMaps(feature_maps={"P3": torch.randn(2, 3, 16, 16)})
    rois = torch.tensor(
        [
            [0, 1, 1, 8, 8],
            [1, 0, 0, 15, 15],
        ],
        dtype=torch.float32,
    )

    out = layer(features, rois)

    assert out.feature_maps["roi"].shape == (2, 3, 7, 7)
    assert out.pooled.shape == (2, 3)


def test_roi_align_accepts_integer_rois_and_casts_them():
    in_spec = FeatureSpec(channels={"P3": 2}, strides={"P3": 1})
    layer = ROIAlign(in_spec=in_spec, cfg=ROIAlignConfig(output_size=4, stage="P3", spatial_scale=1.0))
    features = FeatureMaps(feature_maps={"P3": torch.randn(1, 2, 10, 10)})
    rois = torch.tensor([[0, 1, 1, 9, 9]], dtype=torch.int64)

    out = layer(features, rois)

    assert out.feature_maps["roi"].shape == (1, 2, 4, 4)


def test_roi_align_rejects_invalid_shapes():
    in_spec = FeatureSpec(channels={"P3": 3}, strides={"P3": 1})
    layer = ROIAlign(in_spec=in_spec, cfg=ROIAlignConfig(output_size=4, stage="P3", spatial_scale=1.0))

    with pytest.raises(ValueError, match="feature"):
        layer(
            FeatureMaps(feature_maps={"P3": torch.randn(2, 3, 8)}),
            torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="rois"):
        layer(
            FeatureMaps(feature_maps={"P3": torch.randn(1, 3, 8, 8)}),
            torch.tensor([0, 0, 1, 1], dtype=torch.float32),
        )


def test_roi_align_factory_builds_registered_module():
    model = RegionExtractor.create(
        "roi_align",
        in_spec=FeatureSpec(channels={"P3": 8}, strides={"P3": 4}),
        output_size=5,
        stage="P3",
    )

    assert isinstance(model, ROIAlign)
    assert model.out_spec.embed_dim == 8


FPN_SPEC = FeatureSpec(
    channels={"P2": 2, "P3": 2, "P4": 2, "P5": 2},
    strides={"P2": 4, "P3": 8, "P4": 16, "P5": 32},
)


def _square(size: float, x0: float = 0.0) -> list[float]:
    return [x0, 0.0, x0 + size, size]


def test_assign_boxes_to_levels_follows_the_fpn_formula():
    boxes = torch.tensor([_square(224), _square(112), _square(448), _square(223), _square(16), _square(2000)])

    levels = assign_boxes_to_levels(boxes, min_level=2, max_level=5)

    # 224 -> P4, 112 -> P3, 448 -> P5, just under 224 -> P3, tiny -> clamped to P2, huge -> clamped to P5
    assert (levels + 2).tolist() == [4, 3, 5, 3, 2, 5]


def test_multi_level_roi_align_pools_each_roi_from_its_assigned_level():
    layer = ROIAlign(FPN_SPEC, ROIAlignConfig(stages=("P5", "P3", "P2", "P4"), output_size=2))
    # Every level is constant (value = level number), so the pooled value reveals the level used.
    features = FeatureMaps(feature_maps={
        name: torch.full((2, 2, 512 // stride, 512 // stride), float(level))
        for (name, stride), level in zip(FPN_SPEC.strides.items(), (2, 3, 4, 5))
    })
    rois = torch.tensor([[0.0, *_square(16, 8)], [1.0, *_square(112, 8)], [0.0, *_square(224, 8)], [1.0, *_square(448, 8)]])

    out = layer(features, rois)

    assert layer.stages == ("P2", "P3", "P4", "P5")
    assert out.feature_maps["roi"].shape == (4, 2, 2, 2)
    assert out.pooled[:, 0].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_single_stage_roi_align_uses_only_that_stage():
    layer = ROIAlign(FPN_SPEC, ROIAlignConfig(stage="P3", output_size=2))
    features = FeatureMaps(feature_maps={"P3": torch.full((1, 2, 32, 32), 3.0)})

    out = layer(features, torch.tensor([[0.0, *_square(224)]]))

    assert out.pooled[0].tolist() == [3.0, 3.0]


def test_multi_level_roi_align_rejects_non_contiguous_levels():
    with pytest.raises(ValueError, match="double"):
        ROIAlign(FPN_SPEC, ROIAlignConfig(stages=("P2", "P4")))
    with pytest.raises(ValueError, match="either"):
        ROIAlign(FPN_SPEC, ROIAlignConfig(stage="P2", stages=("P2", "P3")))


def test_roi_align_keeps_roi_coordinates_in_fp32_for_low_precision_features(monkeypatch):
    seen = {}

    def spy(input, boxes, **kwargs):
        seen["input"], seen["boxes"] = input.dtype, boxes.dtype
        return torch.zeros((boxes.shape[0], input.shape[1], 2, 2), dtype=input.dtype)

    monkeypatch.setattr(roi_align_module, "tv_roi_align", spy)
    layer = ROIAlign(FeatureSpec(channels={"P3": 2}, strides={"P3": 1}), ROIAlignConfig(stage="P3", output_size=2))
    features = FeatureMaps(feature_maps={"P3": torch.randn(1, 2, 8, 8, dtype=torch.bfloat16)})

    out = layer(features, torch.tensor([[0.0, 301.3, 1.0, 303.7, 5.0]]))

    assert seen == {"input": torch.float32, "boxes": torch.float32}  # 301.3 would round to 302 in bf16
    assert out.feature_maps["roi"].dtype == torch.bfloat16
