import torch

from optastra.architectures import Architecture
from optastra.architectures.fast_rcnn import FastRCNN
from optastra.core.component_ref import ComponentRef


def test_fast_rcnn_forward_requires_rois_and_returns_box_outputs():
    model = Architecture.create(
        "fast_rcnn_r18_fpn",
        num_classes=5,
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64}),
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
    assert out.logits.shape == (2, 6)
    assert out.values.shape == (2, 4)
    assert "roi_boxes" in out.extra


def test_fast_rcnn_variant_builds_from_registry():
    model = Architecture.create("fast_rcnn_r50_fpn", num_classes=3)
    assert isinstance(model, FastRCNN)