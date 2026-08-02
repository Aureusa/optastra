import torch

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


def test_neck_config_returns_fpn_default_config():
    cfg = Neck.config("fpn")

    assert cfg.out_channels == 256
    assert cfg.preact is False


def test_neck_create_builds_fpn_with_overrides():
    model = Neck.create("fpn", _fpn_in_spec(), out_channels=128)

    assert model.out_spec.channels["P2"] == 128
    assert model.out_spec.channels["P5"] == 128

    out = model(_fpn_inputs())
    assert out.feature_maps["P2"].shape == (2, 128, 56, 56)
    assert out.feature_maps["P3"].shape == (2, 128, 28, 28)
    assert out.feature_maps["P4"].shape == (2, 128, 14, 14)
    assert out.feature_maps["P5"].shape == (2, 128, 7, 7)


def test_neck_describe_prints_fpn_config(capsys):
    Neck.describe("fpn")
    captured = capsys.readouterr()

    assert "fpn:" in captured.out
    assert "out_channels" in captured.out
    assert "preact" in captured.out
