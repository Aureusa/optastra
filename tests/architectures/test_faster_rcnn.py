import torch

from optastra.core.component_ref import ComponentRef
from optastra.architectures import Architecture
from optastra.architectures.faster_rcnn import FasterRCNN
from optastra.detection import keys
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


def _tiny_faster_rcnn():
    torch.manual_seed(0)
    return Architecture.create(
        "faster_rcnn_r18_fpn",
        num_classes=3,
        roi_box_head=ComponentRef("roi_box_head", {"fc_hidden_features": 16}),
    )


def test_faster_rcnn_proposals_respect_each_images_unpadded_size():
    model = _tiny_faster_rcnn()
    out = model(torch.randn(2, 3, 64, 64), image_sizes=[(64, 64), (32, 48)])

    proposals = out.extra[keys.ROI_BOXES]
    second = proposals[proposals[:, 0] == 1]
    assert second.numel() > 0
    assert second[:, [1, 3]].max() <= 48 and second[:, [2, 4]].max() <= 32
    assert out.extra[keys.IMAGE_SIZES] == [(64, 64), (32, 48)]


def test_faster_rcnn_defaults_image_sizes_to_the_batch_size():
    out = _tiny_faster_rcnn()(torch.randn(2, 3, 64, 64))
    assert out.extra[keys.IMAGE_SIZES] == [(64, 64), (64, 64)]


def test_faster_rcnn_appends_gt_boxes_to_the_rois():
    model = _tiny_faster_rcnn()
    rois = torch.tensor([[0.0, 0.0, 0.0, 20.0, 20.0]])
    gt = [torch.tensor([[1.0, 2.0, 30.0, 40.0]]), torch.zeros((0, 4)), ]
    out = model(torch.randn(2, 3, 64, 64), rois=rois, gt_boxes=gt)

    assert torch.equal(out.extra[keys.ROI_BOXES], torch.tensor([[0.0, 0.0, 0.0, 20.0, 20.0], [0.0, 1.0, 2.0, 30.0, 40.0]]))
    assert out.logits.shape == (2, 4)


def test_fpn_presets_use_multi_level_roi_align():
    model = _tiny_faster_rcnn()
    assert model.region_extractor.stages == ("P2", "P3", "P4", "P5")
