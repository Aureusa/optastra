import torch

from optastra.core.component_ref import ComponentRef
from optastra.architectures import Architecture
from optastra.architectures.masked_rcnn import MaskRCNN
from optastra.nn.features import FeatureMaps


def test_masked_rcnn_forward_returns_boxes_and_masks():
    model = Architecture.create(
        "mask_rcnn_r18_fpn",
        num_classes=4,
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
        mask_head=ComponentRef("mask_rcnn_head", {"conv_dims": (32, 32), "upsample_dim": 16}),
        proposal_generator=ComponentRef("rpn", {"num_anchors": 3}),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
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

    assert out.logits.shape == (2, 5)
    assert out.values.shape == (2, 4)
    assert out.masks.shape == (2, 4, 28, 28)
    assert "rpn" in out.extra
    assert isinstance(out.extra["rpn"], FeatureMaps)


def test_masked_rcnn_c5_variant_builds_without_fpn():
    model = Architecture.create("mask_rcnn_r18_c5", num_classes=3)

    assert isinstance(model, MaskRCNN)
    assert model.neck is None
