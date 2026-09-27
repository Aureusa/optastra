import pytest
import torch

from optastra.detection import keys
from optastra.nn.features import FeatureMaps, FeatureSpec
from optastra.nn.blocks.geometry.boxes import clip_boxes_to_image, flatten_anchor_predictions, generate_anchors
from optastra.proposal_generators import ProposalGenerator
from optastra.proposal_generators.rpn import RPN, RPNConfig
from tests.proposal_generators._fixed_rpn import anchor_box, make_fixed_rpn


def test_rpn_forward_single_feature_map_shapes():
    in_spec = FeatureSpec(channels={"P3": 64}, strides={"P3": 8})
    rpn = RPN(in_spec=in_spec, cfg=RPNConfig(num_anchors=3))
    x = FeatureMaps(feature_maps={"P3": torch.randn(2, 64, 32, 32)})

    out = rpn(x)

    assert out.feature_maps["P3_objectness"].shape == (2, 3, 32, 32)
    assert out.feature_maps["P3_deltas"].shape == (2, 12, 32, 32)


def test_rpn_forward_multi_level_shapes():
    in_spec = FeatureSpec(
        channels={"P3": 32, "P4": 32},
        strides={"P3": 8, "P4": 16},
    )
    rpn = RPN(
        in_spec=in_spec,
        cfg=RPNConfig(
            num_anchors=5,
            anchor_scales=(2.0, 4.0, 6.0, 8.0, 10.0),
            aspect_ratios=(1.0,),
            conv_dims=(64, 64),
            in_features=("P3", "P4"),
        ),
    )
    feats = FeatureMaps(
        feature_maps={
            "P3": torch.randn(2, 32, 64, 64),
            "P4": torch.randn(2, 32, 32, 32),
        }
    )

    out = rpn(feats)

    assert out.feature_maps["P3_objectness"].shape == (2, 5, 64, 64)
    assert out.feature_maps["P3_deltas"].shape == (2, 20, 64, 64)
    assert out.feature_maps["P4_objectness"].shape == (2, 5, 32, 32)
    assert out.feature_maps["P4_deltas"].shape == (2, 20, 32, 32)


def test_rpn_rejects_invalid_conv_dims():
    with pytest.raises(ValueError, match="must be > 0"):
        RPN(
            in_spec=FeatureSpec(channels={"P3": 16}, strides={"P3": 8}),
            cfg=RPNConfig(num_anchors=3, conv_dims=(0,)),
        )


def test_rpn_factory_builds_registered_module():
    model = ProposalGenerator.create(
        "rpn",
        in_spec=FeatureSpec(channels={"P3": 64}, strides={"P3": 8}),
        num_anchors=9,
        anchor_scales=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0),
        aspect_ratios=(1.0,),
    )

    assert isinstance(model, RPN)
    assert model.out_spec.channels["P3_objectness"] == 9


def test_clip_boxes_to_image_allows_backpropagation():
    boxes = torch.tensor(
        [[-2.0, -1.0, 12.0, 14.0], [1.0, 2.0, 9.0, 11.0]],
        requires_grad=True,
    )

    clipped = clip_boxes_to_image(boxes, (10, 10))
    loss = clipped.sum()
    loss.backward()

    assert boxes.grad is not None


# --- anchor <-> prediction alignment (hand-built cases) -----------------------


def test_flatten_anchor_predictions_matches_generate_anchors_order():
    num_anchors, h, w = 3, 2, 4
    anchors = generate_anchors(h, w, stride=8, scales=(1.0,), aspect_ratios=(0.5, 1.0, 2.0), device="cpu")
    # Encode (anchor, y, x) in the prediction map the way a conv head lays it out: (N, A*K, H, W).
    pred = torch.zeros(1, num_anchors * 4, h, w)
    for a in range(num_anchors):
        for y in range(h):
            for x in range(w):
                pred[0, a * 4 : (a + 1) * 4, y, x] = torch.tensor([a, y, x, 0.0])

    flat = flatten_anchor_predictions(pred, 4)[0]

    assert flat.shape == (h * w * num_anchors, 4)
    for i, (a, y, x, _) in enumerate(flat.tolist()):
        # Anchor i must be centred on the cell the prediction came from, with that anchor's shape.
        cx, cy = (anchors[i, 0] + anchors[i, 2]) / 2, (anchors[i, 1] + anchors[i, 3]) / 2
        assert cx.item() == pytest.approx((x + 0.5) * 8)
        assert cy.item() == pytest.approx((y + 0.5) * 8)
        assert i % num_anchors == int(a)


def _single_hot_level(h=4, w=4, anchor=2, y=1, x=3, dx=-0.5):
    objectness = torch.full((1, 3, h, w), -10.0)
    objectness[0, anchor, y, x] = 10.0
    deltas = torch.zeros((1, 12, h, w))
    deltas[0, anchor * 4 + 0, y, x] = dx
    return objectness, deltas


def test_rpn_top_scoring_anchor_becomes_top_proposal():
    objectness, deltas = _single_hot_level()
    rpn = make_fixed_rpn({"P3": objectness}, {"P3": deltas}, {"P3": 8}, post_nms_topk=1)
    features = FeatureMaps(feature_maps={"P3": torch.zeros(1, 4, 4, 4)}, extra={keys.IMAGE_SIZES: [(32, 32)]})

    out = rpn(features)

    # Anchor 2 (aspect ratio 2.0) of cell (y=1, x=3): centre (28, 12); dx=-0.5 shifts it by half a width.
    x1, y1, x2, y2 = anchor_box(cx=28.0, cy=12.0, stride=8, ratio=2.0)
    shift = -0.5 * (x2 - x1)
    expected = torch.tensor([[0.0, x1 + shift, y1, x2 + shift, y2]])
    assert torch.allclose(out.extra[keys.PROPOSALS], expected, atol=1e-4)
    assert out.extra[keys.PROPOSAL_SCORES].tolist() == [10.0]


def test_rpn_flattened_outputs_are_anchor_aligned():
    objectness, deltas = _single_hot_level()
    rpn = make_fixed_rpn({"P3": objectness}, {"P3": deltas}, {"P3": 8})
    out = rpn(FeatureMaps(feature_maps={"P3": torch.zeros(1, 4, 4, 4)}))

    best = out.extra[keys.OBJECTNESS_LOGITS][0].argmax()
    assert out.extra[keys.ANCHORS][best].tolist() == pytest.approx(anchor_box(28.0, 12.0, 8, 2.0), abs=1e-4)
    assert out.extra[keys.BBOX_DELTAS][0, best].tolist() == [-0.5, 0.0, 0.0, 0.0]
    assert keys.PROPOSALS not in out.extra  # no image sizes -> no proposal selection


def test_rpn_proposals_are_detached():
    in_spec = FeatureSpec(channels={"P3": 8}, strides={"P3": 8})
    rpn = RPN(in_spec=in_spec, cfg=RPNConfig())
    features = FeatureMaps(feature_maps={"P3": torch.randn(1, 8, 8, 8)}, extra={keys.IMAGE_SIZES: [(64, 64)]})

    out = rpn(features)

    assert out.extra[keys.OBJECTNESS_LOGITS].requires_grad
    assert not out.extra[keys.PROPOSALS].requires_grad
    assert not out.extra[keys.PROPOSAL_SCORES].requires_grad


def test_rpn_clips_proposals_to_each_images_own_size():
    in_spec = FeatureSpec(channels={"P3": 8}, strides={"P3": 8})
    rpn = RPN(in_spec=in_spec, cfg=RPNConfig(anchor_scales=(4.0,)))
    feat = torch.randn(1, 8, 8, 8).expand(2, -1, -1, -1)  # identical predictions for both images
    features = FeatureMaps(feature_maps={"P3": feat}, extra={keys.IMAGE_SIZES: [(64, 64), (24, 40)]})

    proposals = rpn(features).extra[keys.PROPOSALS]

    first, second = proposals[proposals[:, 0] == 0], proposals[proposals[:, 0] == 1]
    assert first[:, [1, 3]].max() > 40  # the full-size image uses its whole width
    assert second[:, [1, 3]].max() <= 40 and second[:, [2, 4]].max() <= 24


def test_rpn_rejects_non_xyxy_box_dim():
    with pytest.raises(ValueError, match="box_dim"):
        RPN(FeatureSpec(channels={"P3": 8}, strides={"P3": 8}), RPNConfig(box_dim=5))
