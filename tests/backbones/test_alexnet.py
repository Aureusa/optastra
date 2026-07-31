import torch

from optastra.backbones.alexnet import alexnet


def test_alexnet_backbone_forward_shape():
    model = alexnet()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert "out" in out.feature_maps
    assert out.feature_maps["out"].shape == (2, 256, 6, 6)
