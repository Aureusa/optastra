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
    default_cfg = Backbone.get_default_config("resnet18")
    assert default_cfg.in_channels == 3

    _ = Backbone.create("resnet18", in_channels=1)

    # dataclasses.replace should create a new config object and keep registry defaults intact
    assert Backbone.get_default_config("resnet18").in_channels == 3


def test_resnet_block_is_a_yaml_safe_string():
    assert Backbone.get_default_config("resnet18").block == "basic"
    assert Backbone.get_default_config("resnet50").block == "bottleneck"

    model = Backbone.create("resnet18", block="bottleneck")
    assert model.out_spec.channels["C5"] == 2048

    with pytest.raises(ValueError, match="Unknown ResNet block"):
        Backbone.create("resnet18", block="BottleneckResidualBlock")


def test_resnet_create_does_not_share_mutable_config_with_registry():
    model = Backbone.create("resnet18")
    model.cfg.layers.append(99)
    assert Backbone.get_default_config("resnet18").layers == [2, 2, 2, 2]
    assert Backbone.create("resnet18").cfg.layers == [2, 2, 2, 2]


def test_preact_resnet_applies_final_bn_relu_to_c5():
    model = Backbone.create("resnet18", preact=True).eval()
    with torch.no_grad():
        # make the final BN a visible affine map so we can check it is applied
        model.final_norm.weight.fill_(2.0)
        model.final_norm.bias.fill_(-0.5)
        x = torch.randn(1, 3, 64, 64)
        out = model(x)

        stage_out = model.stem(x)
        for stage in model.stages:
            stage_out = stage(stage_out)
        expected = torch.relu(model.final_norm(stage_out))

    torch.testing.assert_close(out.feature_maps["C5"], expected)
    assert (out.feature_maps["C5"] >= 0).all()


def test_resnet_uses_kaiming_init_and_identity_batchnorm():
    torch.manual_seed(0)
    model = Backbone.create("resnet50")
    conv = model.stages[2][0].conv2.conv            # 3x3, 256 -> 256
    fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
    expected_std = (2.0 / fan_out) ** 0.5
    assert abs(conv.weight.std().item() - expected_std) / expected_std < 0.05

    bn = model.stages[2][0].conv3.norm
    assert torch.all(bn.weight == 1) and torch.all(bn.bias == 0)


@pytest.mark.parametrize("name", ["resnet18", "resnet50"])
def test_zero_init_residual_makes_blocks_start_as_identity(name):
    model = Backbone.create(name, zero_init_residual=True).eval()
    for stage in model.stages:
        for block in stage:
            assert torch.all(block.last_norm.weight == 0)

    # a block without a projection shortcut then computes relu(x + 0) == x for x >= 0
    block = model.stages[0][1]
    assert block.downsample is None
    x = torch.rand(1, block.last_norm.num_features, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(block(x), x)
