import torch
import pytest

from optastra.architectures import Architecture
from optastra.architectures.faster_rcnn import FasterRCNN


def test_faster_rcnn_forward_returns_head_output_and_rpn_maps():
    model = Architecture.create(
        "faster_rcnn_r18_fpn",
        num_classes=6,
        head_overrides={"hidden_features": 64, "num_layers": 2},
        proposal_generator_overrides={"num_anchors": 3},
        region_extractor_overrides={"stage": "P2", "output_size": 7},
    )

    images = torch.randn(2, 3, 128, 128)
    rois = torch.tensor(
        [
            [0, 0, 0, 64, 64],
            [1, 16, 16, 96, 96],
            [1, 0, 0, 127, 127],
        ],
        dtype=torch.float32,
    )

    out = model(images, rois)

    assert out.logits.shape == (3, 6)
    assert out.values.shape == (3, 4)
    assert "rpn" in out.extra
    assert "P2_objectness" in out.extra["rpn"]
    assert "P2_deltas" in out.extra["rpn"]


def test_faster_rcnn_raises_without_rois_or_proposals():
    model = Architecture.create(
        "faster_rcnn_r18_fpn",
        num_classes=3,
        region_extractor_overrides={"stage": "P2", "output_size": 7},
    )

    images = torch.randn(2, 3, 128, 128)

    with pytest.raises(ValueError, match="requires proposal boxes"):
        model(images)


def test_registered_c5_variant_builds_without_fpn():
    model = Architecture.create("faster_rcnn_r18_c5", num_classes=4)

    assert isinstance(model, FasterRCNN)
    assert model.neck is None
