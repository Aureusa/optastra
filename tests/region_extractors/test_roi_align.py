import pytest
import torch

from optastra.nn.features import FeatureMaps, FeatureSpec
from optastra.region_extractors import RegionExtractor
from optastra.region_extractors.roi_align import ROIAlign, ROIAlignConfig


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
