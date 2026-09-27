import torch
import pytest

from optastra.backbones import Backbone


def test_vgg11_forward_shape():
    model = Backbone.create("vgg11")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    # every stage ends with a 2x2 max-pool: C_k has stride 2**k, widths cap at 512
    assert sorted(out.feature_maps.keys()) == ["C1", "C2", "C3", "C4", "C5"]
    assert out.feature_maps["C1"].shape == (2, 64, 112, 112)
    assert out.feature_maps["C2"].shape == (2, 128, 56, 56)
    assert out.feature_maps["C3"].shape == (2, 256, 28, 28)
    assert out.feature_maps["C4"].shape == (2, 512, 14, 14)
    assert out.feature_maps["C5"].shape == (2, 512, 7, 7)
    assert model.out_spec.strides == {"C1": 2, "C2": 4, "C3": 8, "C4": 16, "C5": 32}


def test_vgg16_forward_shape():
    model = Backbone.create("vgg16")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C1", "C2", "C3", "C4", "C5"]
    assert out.feature_maps["C1"].shape == (2, 64, 112, 112)
    assert out.feature_maps["C2"].shape == (2, 128, 56, 56)
    assert out.feature_maps["C3"].shape == (2, 256, 28, 28)
    assert out.feature_maps["C4"].shape == (2, 512, 14, 14)
    assert out.feature_maps["C5"].shape == (2, 512, 7, 7)   # == torchvision vgg16_bn.features output


def test_vgg16_parameter_count_matches_reference():
    # torchvision vgg16_bn.features has 14,723,136 parameters; our convs are followed by BN
    # and therefore have no bias, which removes one parameter per output channel.
    model = Backbone.create("vgg16")
    conv_biases = 64 * 2 + 128 * 2 + 256 * 3 + 512 * 3 + 512 * 3
    assert sum(p.numel() for p in model.parameters()) == 14_723_136 - conv_biases


def test_vgg_create_with_overrides_applies_config():
    model = Backbone.create("vgg11", in_channels=1, stem_channels=32, preact=True)

    assert model.cfg.in_channels == 1
    assert model.cfg.stem_channels == 32
    assert model.cfg.preact is True
    assert model.out_spec.channels == {"C1": 32, "C2": 64, "C3": 128, "C4": 256, "C5": 512}

    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.feature_maps["C5"].shape == (2, 512, 7, 7)


def test_vgg_max_channels_caps_stage_width():
    model = Backbone.create("vgg11", stem_channels=64, max_channels=256)
    assert model.out_spec.channels == {"C1": 64, "C2": 128, "C3": 256, "C4": 256, "C5": 256}


def test_vgg_create_with_unknown_override_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Backbone.create("vgg11", not_a_real_field=123)


def test_vgg_create_override_does_not_mutate_default_config():
    default_cfg = Backbone.get_default_config("vgg11")
    assert default_cfg.in_channels == 3

    _ = Backbone.create("vgg11", in_channels=1)

    assert Backbone.get_default_config("vgg11").in_channels == 3
