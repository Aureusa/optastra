import torch

from optastra.backbones.resnet import resnet18, resnet50


def test_resnet18_forward_shape():
    model = resnet18()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C2", "C3", "C4", "C5"]
    assert out.feature_maps["C2"].shape == (2, 64, 56, 56)
    assert out.feature_maps["C3"].shape == (2, 128, 28, 28)
    assert out.feature_maps["C4"].shape == (2, 256, 14, 14)
    assert out.feature_maps["C5"].shape == (2, 512, 7, 7)


def test_resnet50_forward_shape():
    model = resnet50()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C2", "C3", "C4", "C5"]
    assert out.feature_maps["C2"].shape == (2, 256, 56, 56)
    assert out.feature_maps["C3"].shape == (2, 512, 28, 28)
    assert out.feature_maps["C4"].shape == (2, 1024, 14, 14)
    assert out.feature_maps["C5"].shape == (2, 2048, 7, 7)
