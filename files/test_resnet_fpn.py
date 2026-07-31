import torch

from vision.backbones.resnet import resnet18, resnet50
from vision.necks.fpn import FPN


def test_resnet18_stage_shapes():
    model = resnet18()
    x = torch.randn(2, 3, 224, 224)
    feats = model(x)
    expected_strides = {"C2": 4, "C3": 8, "C4": 16, "C5": 32}
    for name, stride in expected_strides.items():
        expected_hw = 224 // stride
        assert feats.feature_maps[name].shape[-2:] == (expected_hw, expected_hw)
        assert feats.feature_maps[name].shape[1] == model.out_channels[name]
    print("resnet18 stage shapes OK:", {k: tuple(v.shape) for k, v in feats.feature_maps.items()})


def test_resnet50_fpn():
    backbone = resnet50()
    neck = FPN(in_channels=backbone.out_channels, out_channels=256)

    x = torch.randn(2, 3, 224, 224)
    feats = backbone(x)
    pyramid = neck(feats)

    for name in ["P2", "P3", "P4", "P5"]:
        assert pyramid[name].shape[1] == 256
    # all pyramid levels should match the spatial size of their source stage
    assert pyramid["P2"].shape[-2:] == feats.feature_maps["C2"].shape[-2:]
    assert pyramid["P5"].shape[-2:] == feats.feature_maps["C5"].shape[-2:]
    print("resnet50 + FPN OK:", {k: tuple(v.shape) for k, v in pyramid.items()})


if __name__ == "__main__":
    test_resnet18_stage_shapes()
    test_resnet50_fpn()
    print("all tests passed")
