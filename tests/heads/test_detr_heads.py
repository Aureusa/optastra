import math

import pytest
import torch

from optastra.decoders import keys as decoder_keys
from optastra.detection import keys
from optastra.heads import DETRBoxHead, Head, KeypointHead
from optastra.nn.features import FeatureMaps, FeatureSpec


D = 16


def _tokens(B=2, M=5, refs=None):
    torch.manual_seed(1)
    extra = {decoder_keys.REFERENCE_POINTS: refs} if refs is not None else {}
    return FeatureMaps(patch_tokens=torch.randn(B, M, D), extra=extra)


def _refs(B=2, M=5):
    g = torch.Generator().manual_seed(2)
    return torch.rand(B, M, 2, generator=g) * 0.9 + 0.05


def test_heads_registered():
    names = Head.list_all()
    assert "keypoint_head" in names and "detr_box_head" in names


@pytest.mark.parametrize("name", ["keypoint_head", "detr_box_head"])
def test_heads_reject_in_spec_without_embed_dim(name):
    with pytest.raises(ValueError, match="embed_dim"):
        Head.create(name, FeatureSpec(channels={"C5": 16}, strides={"C5": 32}))


@pytest.mark.parametrize("name", ["keypoint_head", "detr_box_head"])
def test_heads_reject_features_without_tokens(name):
    head = Head.create(name, FeatureSpec(embed_dim=D))
    with pytest.raises(ValueError, match="patch_tokens"):
        head(FeatureMaps(feature_maps={"C5": torch.zeros(1, D, 2, 2)}))


# --- keypoint_head ------------------------------------------------------------------------

def test_keypoint_head_construction_and_shapes():
    head = Head.create("keypoint_head", FeatureSpec(embed_dim=D), hidden_features=32)
    assert isinstance(head, KeypointHead)
    assert head.out_spec.embed_dim == 2
    out = head(_tokens())
    assert out.scores.shape == (2, 5)
    assert out.extra[keys.POINTS].shape == (2, 5, 2)
    assert keys.POINT_LOG_SCALES not in out.extra


def test_keypoint_head_starts_exactly_at_reference_points():
    head = Head.create("keypoint_head", FeatureSpec(embed_dim=D), hidden_features=32)
    refs = _refs()
    pts = head(_tokens(refs=refs)).extra[keys.POINTS]
    torch.testing.assert_close(pts, refs, rtol=0, atol=1e-6)


def test_keypoint_head_without_reference_starts_at_centre():
    head = Head.create("keypoint_head", FeatureSpec(embed_dim=D), hidden_features=32)
    pts = head(_tokens()).extra[keys.POINTS]
    assert torch.equal(pts, torch.full((2, 5, 2), 0.5))


def test_keypoint_head_points_are_offsets_in_logit_space():
    head = Head.create("keypoint_head", FeatureSpec(embed_dim=D), hidden_features=32)
    with torch.no_grad():
        head.location.mlp[-1].bias.copy_(torch.tensor([1.0, -2.0]))
    refs = _refs()
    pts = head(_tokens(refs=refs)).extra[keys.POINTS]
    expected = torch.sigmoid(torch.logit(refs) + torch.tensor([1.0, -2.0]))
    torch.testing.assert_close(pts, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prior_prob", [0.1, 0.01])
def test_keypoint_head_objectness_bias_is_logit_of_prior(prior_prob):
    head = Head.create("keypoint_head", FeatureSpec(embed_dim=D), prior_prob=prior_prob)
    assert head.objectness.bias.item() == pytest.approx(math.log(prior_prob / (1 - prior_prob)), rel=1e-6)
    with torch.no_grad():
        head.objectness.weight.zero_()
    probs = head(_tokens()).scores.sigmoid()
    torch.testing.assert_close(probs, torch.full((2, 5), prior_prob))


def test_keypoint_head_predict_uncertainty_adds_log_scales():
    head = Head.create("keypoint_head", FeatureSpec(embed_dim=D), predict_uncertainty=True)
    refs = _refs()
    out = head(_tokens(refs=refs))
    assert out.extra[keys.POINT_LOG_SCALES].shape == (2, 5, 2)
    assert torch.equal(out.extra[keys.POINT_LOG_SCALES], torch.zeros(2, 5, 2))   # zero-init: scale 1
    torch.testing.assert_close(out.extra[keys.POINTS], refs, rtol=0, atol=1e-6)
    with torch.no_grad():
        head.location.mlp[-1].bias.copy_(torch.tensor([0.0, 0.0, -1.0, 2.0]))
    out = head(_tokens(refs=refs))
    torch.testing.assert_close(out.extra[keys.POINT_LOG_SCALES], torch.tensor([-1.0, 2.0]).expand(2, 5, 2))
    torch.testing.assert_close(out.extra[keys.POINTS], refs, rtol=0, atol=1e-6)


def test_keypoint_head_gradients_flow_through_zero_init_offset():
    head = Head.create("keypoint_head", FeatureSpec(embed_dim=D))
    out = head(_tokens(refs=_refs()))
    (out.extra[keys.POINTS].sum() + out.scores.sum()).backward()
    assert head.location.mlp[-1].weight.grad.abs().sum() > 0


# --- detr_box_head -----------------------------------------------------------------------------

def test_detr_box_head_shapes_and_ranges():
    head = Head.create("detr_box_head", FeatureSpec(embed_dim=D), num_classes=3, hidden_features=32)
    assert isinstance(head, DETRBoxHead)
    assert head.out_spec.embed_dim == 4
    out = head(_tokens())
    assert out.logits.shape == (2, 5, 4)          # K + 1, last = no object
    assert out.boxes.shape == (2, 5, 4)
    assert ((out.boxes > 0) & (out.boxes < 1)).all()


def test_detr_box_head_values_without_reference_points():
    head = Head.create("detr_box_head", FeatureSpec(embed_dim=D), num_classes=3, hidden_features=32)
    tokens = _tokens()
    raw = head.bbox_embed(tokens.patch_tokens)
    out = head(tokens)
    torch.testing.assert_close(out.boxes, raw.sigmoid())
    torch.testing.assert_close(out.logits, head.class_embed(tokens.patch_tokens))


def test_detr_box_head_centre_is_offset_from_reference_points():
    head = Head.create("detr_box_head", FeatureSpec(embed_dim=D), num_classes=3, hidden_features=32)
    with torch.no_grad():
        head.bbox_embed.mlp[-1].weight.zero_()
        head.bbox_embed.mlp[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, math.log(3.0)]))
    refs = _refs()
    boxes = head(_tokens(refs=refs)).boxes
    torch.testing.assert_close(boxes[..., :2], refs, rtol=0, atol=1e-6)          # centre = reference
    torch.testing.assert_close(boxes[..., 2], torch.full((2, 5), 0.5))           # w = sigmoid(0)
    torch.testing.assert_close(boxes[..., 3], torch.full((2, 5), 0.75))          # h = sigmoid(log 3)

    with torch.no_grad():
        head.bbox_embed.mlp[-1].bias.copy_(torch.tensor([0.5, -0.5, 0.0, 0.0]))
    boxes = head(_tokens(refs=refs)).boxes
    torch.testing.assert_close(boxes[..., :2], torch.sigmoid(torch.logit(refs) + torch.tensor([0.5, -0.5])),
                               rtol=1e-5, atol=1e-6)
