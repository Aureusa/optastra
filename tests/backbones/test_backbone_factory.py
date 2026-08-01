import torch

from optastra.backbones import Backbone


def test_backbone_config_returns_resnet50_default_config():
    cfg = Backbone.config("resnet50")

    assert cfg.in_channels == 3
    assert cfg.stem_channels == 64
    assert cfg.preact is False
    assert cfg.layers == [3, 4, 6, 3]


def test_backbone_create_builds_resnet50_with_overrides():
    model = Backbone.create("resnet50", in_channels=1)

    assert model.cfg.in_channels == 1
    assert model.cfg.layers == [3, 4, 6, 3]

    x = torch.randn(2, 1, 224, 224)
    out = model(x)
    assert out.feature_maps["C5"].shape == (2, 2048, 7, 7)


def test_backbone_describe_prints_resnet50_config(capsys):
    Backbone.describe("resnet50")
    captured = capsys.readouterr()

    assert "resnet50:" in captured.out
    assert "in_channels" in captured.out
    assert "stem_channels" in captured.out
    assert "preact" in captured.out
    assert "layers" in captured.out
