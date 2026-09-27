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


def test_fpn_handles_backbone_features_from_odd_input_sizes():
    backbone = Backbone.create("resnet50")
    images = torch.randn(2, 3, 300, 300)
    backbone_features = backbone(images)

    neck = Neck.create("fpn", backbone.out_spec, out_channels=128)
    out = neck(backbone_features)

    assert out.feature_maps["P2"].shape[-2:] == (75, 75)
    assert out.feature_maps["P3"].shape[-2:] == (38, 38)
    assert out.feature_maps["P4"].shape[-2:] == (19, 19)
    assert out.feature_maps["P5"].shape[-2:] == (10, 10)



def test_fpn_out_strides_come_from_in_spec():
    # VGG-style C1-C5 at strides 2..32: P-levels must keep those strides, not assume C2 == 4
    in_spec = FeatureSpec(
        channels={f"C{i}": 16 for i in range(1, 6)},
        strides={f"C{i}": 2 ** i for i in range(1, 6)},
    )
    neck = Neck.create("fpn", in_spec, out_channels=8)
    assert neck.out_spec.strides == {"P1": 2, "P2": 4, "P3": 8, "P4": 16, "P5": 32}


def test_fpn_orders_stages_by_stride_not_by_name():
    # lexicographic order would be C10, C8, C9 -- the top-down pass must go C10 -> C9 -> C8
    in_spec = FeatureSpec(
        channels={"C8": 4, "C9": 4, "C10": 4},
        strides={"C8": 256, "C9": 512, "C10": 1024},
    )
    neck = Neck.create("fpn", in_spec, out_channels=4)
    assert neck.stage_names == ["C8", "C9", "C10"]
    assert neck.out_spec.strides == {"P8": 256, "P9": 512, "P10": 1024}

    out = neck(FeatureMaps(feature_maps={
        "C8": torch.randn(1, 4, 8, 8), "C9": torch.randn(1, 4, 4, 4), "C10": torch.randn(1, 4, 2, 2),
    }))
    assert out.feature_maps["P8"].shape == (1, 4, 8, 8)
    assert out.feature_maps["P10"].shape == (1, 4, 2, 2)


def test_fpn_top_down_sum_values():
    # identity-like laterals/outputs make the top-down pathway directly checkable
    in_spec = FeatureSpec(channels={"C2": 1, "C3": 1}, strides={"C2": 4, "C3": 8})
    neck = Neck.create("fpn", in_spec, out_channels=1)
    with torch.no_grad():
        for conv in list(neck.laterals.values()):
            conv.conv.weight.fill_(1.0)
            conv.conv.bias.zero_()
        for conv in neck.outputs.values():
            conv.conv.weight.zero_()
            conv.conv.weight[..., 1, 1] = 1.0
            conv.conv.bias.zero_()

    c2 = torch.arange(16.0).reshape(1, 1, 4, 4)
    c3 = torch.tensor([[[[100.0, 200.0], [300.0, 400.0]]]])
    out = neck(FeatureMaps(feature_maps={"C2": c2, "C3": c3}))

    torch.testing.assert_close(out.feature_maps["P3"], c3)
    upsampled = c3.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    torch.testing.assert_close(out.feature_maps["P2"], c2 + upsampled)
