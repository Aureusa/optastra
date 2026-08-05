import torch
import pytest

from optastra.backbones import Backbone


def test_alexnet_backbone_forward_shape():
    model = Backbone.create("alexnet")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert "out" in out.feature_maps
    assert out.feature_maps["out"].shape == (2, 256, 6, 6)


def test_alexnet_create_with_overrides_applies_config():
    model = Backbone.create("alexnet", in_channels=1, channels=[32, 64, 128, 128, 128])

    assert model.cfg.in_channels == 1
    assert model.cfg.channels == [32, 64, 128, 128, 128]

    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.feature_maps["out"].shape == (2, 128, 6, 6)


def test_alexnet_create_with_unknown_override_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Backbone.create("alexnet", not_a_real_field=123)


def test_alexnet_create_override_does_not_mutate_default_config():
    default_cfg = Backbone.get_default_config("alexnet")
    assert default_cfg.in_channels == 3
    assert default_cfg.channels == [96, 256, 384, 256, 256]

    _ = Backbone.create("alexnet", in_channels=1)

    assert Backbone.get_default_config("alexnet").in_channels == 3
