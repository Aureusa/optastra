import torch

from optastra.core.component_ref import ComponentRef
from optastra.architectures import Architecture
from optastra.architectures.faster_rcnn import FasterRCNN
from optastra.nn.features import FeatureMaps


def test_faster_rcnn_forward_returns_head_output_and_rpn_maps():
    model = Architecture.create(
        "faster_rcnn_r18_fpn",
        num_classes=6,
        proposal_generator=ComponentRef("rpn", {"anchor_scales": (0.5,), "aspect_ratios": [0.5, 1.0, 2.0]}),
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 64})
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

    assert out.logits.shape == (3, 7)
    assert out.values.shape == (3, 4)
    assert "rpn" in out.extra
    assert isinstance(out.extra["rpn"], FeatureMaps)
    assert "P2_objectness" in out.extra["rpn"].feature_maps
    assert "P2_deltas" in out.extra["rpn"].feature_maps
    assert "roi_boxes" in out.extra


def test_faster_rcnn_generates_proposals_when_rois_are_not_given():
    model = Architecture.create(
        "faster_rcnn_r18_fpn",
        num_classes=3,
        region_extractor=ComponentRef("roi_align", {"stage": "P2", "output_size": 7}),
    )

    images = torch.randn(2, 3, 128, 128)
    out = model(images)

    assert out.logits is not None
    assert out.values is not None
    assert out.extra["roi_boxes"].shape[1] == 5


def test_registered_c5_variant_builds_without_fpn():
    model = Architecture.create("faster_rcnn_r18_c5", num_classes=4)

    assert isinstance(model, FasterRCNN)
    assert model.neck is None
