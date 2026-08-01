import torch
import pytest

from optastra.backbones import Backbone


def test_vgg11_forward_shape():
    model = Backbone.create("vgg11")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C1", "C2", "C3", "C4", "C5"]
    assert out.feature_maps["C1"].shape == (2, 64, 224, 224)
    assert out.feature_maps["C2"].shape == (2, 128, 112, 112)
    assert out.feature_maps["C3"].shape == (2, 256, 56, 56)
    assert out.feature_maps["C4"].shape == (2, 512, 28, 28)
    assert out.feature_maps["C5"].shape == (2, 1024, 14, 14)


def test_vgg16_forward_shape():
    model = Backbone.create("vgg16")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C1", "C2", "C3", "C4", "C5"]
    assert out.feature_maps["C1"].shape == (2, 64, 224, 224)
    assert out.feature_maps["C2"].shape == (2, 128, 112, 112)
    assert out.feature_maps["C3"].shape == (2, 256, 56, 56)
    assert out.feature_maps["C4"].shape == (2, 512, 28, 28)
    assert out.feature_maps["C5"].shape == (2, 1024, 14, 14)


def test_vgg_create_with_overrides_applies_config():
    model = Backbone.create("vgg11", in_channels=1, stem_channels=32, preact=True)

    assert model.cfg.in_channels == 1
    assert model.cfg.stem_channels == 32
    assert model.cfg.preact is True

    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.feature_maps["C5"].shape == (2, 512, 14, 14)


def test_vgg_create_with_unknown_override_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Backbone.create("vgg11", not_a_real_field=123)


def test_vgg_create_override_does_not_mutate_default_config():
    default_cfg = Backbone.config("vgg11")
    assert default_cfg.in_channels == 3

    _ = Backbone.create("vgg11", in_channels=1)

    assert Backbone.config("vgg11").in_channels == 3
