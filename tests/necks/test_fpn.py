import torch
import pytest

from optastra.backbones import Backbone
from optastra.necks import Neck
from optastra.nn.features import FeatureMaps, FeatureSpec


def _fpn_in_spec() -> FeatureSpec:
    return FeatureSpec(
        channels={"C2": 256, "C3": 512, "C4": 1024, "C5": 2048},
        strides={"C2": 4, "C3": 8, "C4": 16, "C5": 32},
    )


def _fpn_inputs() -> FeatureMaps:
    return FeatureMaps(
        feature_maps={
            "C2": torch.randn(2, 256, 56, 56),
            "C3": torch.randn(2, 512, 28, 28),
            "C4": torch.randn(2, 1024, 14, 14),
            "C5": torch.randn(2, 2048, 7, 7),
        }
    )


def test_fpn_forward_shape():
    neck = Neck.create("fpn", _fpn_in_spec())
    out = neck(_fpn_inputs())

    assert sorted(out.feature_maps.keys()) == ["P2", "P3", "P4", "P5"]
    assert out.feature_maps["P2"].shape == (2, 256, 56, 56)
    assert out.feature_maps["P3"].shape == (2, 256, 28, 28)
    assert out.feature_maps["P4"].shape == (2, 256, 14, 14)
    assert out.feature_maps["P5"].shape == (2, 256, 7, 7)


def test_fpn_wires_with_backbone_features():
    backbone = Backbone.create("resnet50")
    images = torch.randn(2, 3, 224, 224)
    backbone_features = backbone(images)

    neck = Neck.create("fpn", backbone.out_spec, out_channels=128)
    out = neck(backbone_features)

    assert sorted(out.feature_maps.keys()) == ["P2", "P3", "P4", "P5"]
    assert out.feature_maps["P2"].shape == (2, 128, 56, 56)
    assert out.feature_maps["P3"].shape == (2, 128, 28, 28)
    assert out.feature_maps["P4"].shape == (2, 128, 14, 14)
    assert out.feature_maps["P5"].shape == (2, 128, 7, 7)

