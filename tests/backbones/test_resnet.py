import torch
import pytest

from optastra.backbones import Backbone

def test_resnet18_forward_shape():
    model = Backbone.create("resnet18")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C2", "C3", "C4", "C5"]
    assert out.feature_maps["C2"].shape == (2, 64, 56, 56)
    assert out.feature_maps["C3"].shape == (2, 128, 28, 28)
    assert out.feature_maps["C4"].shape == (2, 256, 14, 14)
    assert out.feature_maps["C5"].shape == (2, 512, 7, 7)


def test_resnet50_forward_shape():
    model = Backbone.create("resnet50")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C2", "C3", "C4", "C5"]
    assert out.feature_maps["C2"].shape == (2, 256, 56, 56)
    assert out.feature_maps["C3"].shape == (2, 512, 28, 28)
    assert out.feature_maps["C4"].shape == (2, 1024, 14, 14)
    assert out.feature_maps["C5"].shape == (2, 2048, 7, 7)


def test_resnet_create_with_overrides_applies_config():
    model = Backbone.create("resnet18", in_channels=1, stem_channels=32, preact=True)

    assert model.cfg.in_channels == 1
    assert model.cfg.stem_channels == 32
    assert model.cfg.preact is True

    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.feature_maps["C5"].shape == (2, 512, 7, 7)


def test_resnet_create_with_unknown_override_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Backbone.create("resnet18", not_a_real_field=123)


def test_resnet_create_override_does_not_mutate_default_config():
    default_cfg = Backbone.config("resnet18")
    assert default_cfg.in_channels == 3

    _ = Backbone.create("resnet18", in_channels=1)

    # dataclasses.replace should create a new config object and keep registry defaults intact
    assert Backbone.config("resnet18").in_channels == 3
