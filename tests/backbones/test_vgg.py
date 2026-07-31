import torch

from optastra.backbones.vgg import vgg11, vgg16


def test_vgg11_forward_shape():
    model = vgg11()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C1", "C2", "C3", "C4", "C5"]
    assert out.feature_maps["C1"].shape == (2, 64, 224, 224)
    assert out.feature_maps["C2"].shape == (2, 128, 112, 112)
    assert out.feature_maps["C3"].shape == (2, 256, 56, 56)
    assert out.feature_maps["C4"].shape == (2, 512, 28, 28)
    assert out.feature_maps["C5"].shape == (2, 1024, 14, 14)


def test_vgg16_forward_shape():
    model = vgg16()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert sorted(out.feature_maps.keys()) == ["C1", "C2", "C3", "C4", "C5"]
    assert out.feature_maps["C1"].shape == (2, 64, 224, 224)
    assert out.feature_maps["C2"].shape == (2, 128, 112, 112)
    assert out.feature_maps["C3"].shape == (2, 256, 56, 56)
    assert out.feature_maps["C4"].shape == (2, 512, 28, 28)
    assert out.feature_maps["C5"].shape == (2, 1024, 14, 14)
