import torch

from optastra.architectures import Architecture
from optastra.architectures.masked_rcnn import MaskRCNN


def test_masked_rcnn_forward_returns_boxes_and_masks():
    model = Architecture.create(
        "mask_rcnn_r18_fpn",
        num_classes=4,
        head_overrides={"hidden_features": 64, "num_layers": 2},
        box_head_overrides={"hidden_features": 64, "num_layers": 2},
        mask_head_overrides={"conv_dims": (32, 32), "upsample_dim": 16},
        proposal_generator_overrides={"num_anchors": 3},
        region_extractor_overrides={"stage": "P2", "output_size": 7},
    )

    images = torch.randn(2, 3, 128, 128)
    rois = torch.tensor(
        [
            [0, 0, 0, 64, 64],
            [1, 16, 16, 96, 96],
        ],
        dtype=torch.float32,
    )

    out = model(images, rois)

    assert out.logits.shape == (2, 4)
    assert out.values.shape == (2, 4)
    assert out.masks.shape == (2, 4, 14, 14)
    assert "rpn" in out.extra


def test_masked_rcnn_c5_variant_builds_without_fpn():
    model = Architecture.create("mask_rcnn_r18_c5", num_classes=3)

    assert isinstance(model, MaskRCNN)
    assert model.neck is None
