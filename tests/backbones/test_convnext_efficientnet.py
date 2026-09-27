import pytest
import torch

from optastra.backbones import Backbone, EfficientNet
from optastra.backbones.efficientnet import EFFICIENTNET_VARIANTS


def test_convnext_uses_truncated_normal_init_with_zero_bias():
    torch.manual_seed(0)
    model = Backbone.create("convnext_tiny")
    linear = model.stages[2][0].pwconv1          # 384 -> 1536
    assert abs(linear.weight.std().item() - 0.02) < 0.002
    assert torch.all(linear.bias == 0)
    assert torch.all(model.stem.conv.bias == 0)


def test_convnext_forward_shapes():
    model = Backbone.create("convnext_tiny").eval()
    out = model(torch.randn(1, 3, 64, 64))
    assert {k: tuple(v.shape) for k, v in out.feature_maps.items()} == {
        "C2": (1, 96, 16, 16), "C3": (1, 192, 8, 8), "C4": (1, 384, 4, 4), "C5": (1, 768, 2, 2),
    }


@pytest.mark.parametrize(
    "name, width, depth",
    [
        ("efficientnet_b0", 1.0, 1.0),
        ("efficientnet_b1", 1.0, 1.1),
        ("efficientnet_b2", 1.1, 1.2),
        ("efficientnet_b3", 1.2, 1.4),
        ("efficientnet_b4", 1.4, 1.8),
        ("efficientnet_b5", 1.6, 2.2),
        ("efficientnet_b6", 1.8, 2.6),
        ("efficientnet_b7", 2.0, 3.1),
    ],
)
def test_efficientnet_variants_use_official_multipliers(name, width, depth):
    cfg = Backbone.get_default_config(name)
    assert (cfg.width_mult, cfg.depth_mult) == (width, depth)


def test_efficientnet_variants_are_registered_from_the_table():
    assert sorted(EFFICIENTNET_VARIANTS) == Backbone.list_all(filter="efficientnet")
    assert isinstance(Backbone.create("efficientnet_b0"), EfficientNet)


def test_efficientnet_b0_levels_match_timm_feature_channels():
    # timm efficientnet_b0 features: 24 @ /4, 40 @ /8, 112 @ /16, 320 @ /32
    model = Backbone.create("efficientnet_b0")
    assert model.out_spec.channels == {"C2": 24, "C3": 40, "C4": 112, "C5": 320}
    assert model.out_spec.strides == {"C2": 4, "C3": 8, "C4": 16, "C5": 32}


def test_efficientnet_b4_widths():
    # width 1.4: channels round to multiples of 8 -> 32, 56, 160, 448
    model = Backbone.create("efficientnet_b4")
    assert model.out_spec.channels == {"C2": 32, "C3": 56, "C4": 160, "C5": 448}
