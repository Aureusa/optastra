"""Set prediction (DETR family): focal loss, Hungarian matching, set criteria, postprocessors."""
import itertools
import math

import pytest
import torch
import torch.nn.functional as F
import torchvision

from optastra.core.component_ref import ComponentRef
from optastra.detection import DetectionCriterion, Matcher, Postprocessor, keys
from optastra.detection.criteria.set_prediction import point_set_losses
from optastra.detection.matching.hungarian import HungarianMatcher, HungarianMatcherConfig, linear_assignment
from optastra.losses import sigmoid_focal_loss
from optastra.nn.features import HeadOutput


# --- sigmoid_focal_loss ---------------------------------------------------------------------

@pytest.mark.parametrize("alpha, gamma", [(0.25, 2.0), (-1.0, 2.0), (0.5, 0.0), (0.75, 1.5)])
def test_sigmoid_focal_loss_matches_torchvision(alpha, gamma):
    torch.manual_seed(0)
    logits = torch.randn(4, 7) * 3
    targets = (torch.rand(4, 7) > 0.5).float()
    expected = torchvision.ops.sigmoid_focal_loss(logits, targets, alpha=alpha, gamma=gamma, reduction="none")
    torch.testing.assert_close(sigmoid_focal_loss(logits, targets, alpha, gamma), expected)


def test_sigmoid_focal_loss_hand_value():
    # logit ln 3: p = 0.75. positive: 0.25 * 0.25^2 * -ln 0.75; negative: 0.75 * 0.75^2 * -ln 0.25
    x = torch.tensor([math.log(3.0), math.log(3.0)])
    out = sigmoid_focal_loss(x, torch.tensor([1.0, 0.0]))
    assert out[0].item() == pytest.approx(0.25 * 0.0625 * -math.log(0.75), rel=1e-5)
    assert out[1].item() == pytest.approx(0.75 * 0.5625 * -math.log(0.25), rel=1e-5)


# --- linear_assignment ----------------------------------------------------------------------

def _brute_force_min(cost: torch.Tensor) -> float:
    R, C = cost.shape
    if R > C:
        return _brute_force_min(cost.T)
    return min(sum(cost[r, c].item() for r, c in zip(range(R), cols))
               for cols in itertools.permutations(range(C), R))


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (4, 4), (2, 5), (5, 3), (4, 6)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_assignment_is_optimal(shape, seed):
    g = torch.Generator().manual_seed(seed)
    cost = torch.rand(*shape, generator=g) * 10 - 3
    rows, cols = linear_assignment(cost)
    assert rows.dtype == torch.long and cols.dtype == torch.long
    assert len(rows) == min(shape)
    assert len(set(rows.tolist())) == len(rows) and len(set(cols.tolist())) == len(cols)
    assert cost[rows, cols].sum().item() == pytest.approx(_brute_force_min(cost), abs=1e-5)


def test_linear_assignment_empty():
    for shape in [(0, 3), (3, 0), (0, 0)]:
        rows, cols = linear_assignment(torch.zeros(shape))
        assert rows.numel() == 0 and cols.numel() == 0 and rows.dtype == torch.long


# --- HungarianMatcher --------------------------------------------------------------------------

def test_hungarian_matcher_is_registered_and_refuses_anchor_matching():
    assert "hungarian_matcher" in Matcher.list_all()
    m = Matcher.create("hungarian_matcher", cost_bbox=3.0)
    assert isinstance(m, HungarianMatcher) and m.cfg.cost_bbox == 3.0
    with pytest.raises(NotImplementedError):
        m.match(torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1))


def test_box_cost_hand_values():
    m = HungarianMatcher()   # class 1, bbox 5, giou 2
    logits = torch.tensor([[0.0, math.log(3.0)], [math.log(3.0), 0.0]])     # p(class 0) = 0.25, 0.75
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])
    gt_boxes = torch.tensor([[0.6, 0.5, 0.2, 0.2]])
    cost = m.box_cost(logits, boxes, torch.tensor([0]), gt_boxes)
    # pred 0: L1 0.1, IoU = 0.02 / 0.06 = 1/3, enclosing box = union -> GIoU 1/3
    c0 = 5 * 0.1 - 0.25 - 2 * (1 / 3)
    # pred 1: L1 0.4 + 0.3 + 0.1 + 0.1, disjoint, enclosing box 0.55 x 0.45: GIoU = -(0.2475 - 0.05) / 0.2475
    giou1 = -(0.55 * 0.45 - 0.05) / (0.55 * 0.45)
    c1 = 5 * (0.4 + 0.3 + 0.1 + 0.1) - 0.75 - 2 * giou1
    torch.testing.assert_close(cost, torch.tensor([[c0], [c1]]), rtol=1e-5, atol=1e-5)
    pi, gi = m.match_boxes(logits, boxes, torch.tensor([0]), gt_boxes)
    assert pi.tolist() == [0] and gi.tolist() == [0]


def test_point_cost_hand_values_and_matching():
    m = HungarianMatcher(HungarianMatcherConfig(cost_point=1.0, cost_objectness=2.0, point_scale=2.0))
    points = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    objectness = torch.tensor([0.0, math.log(3.0)])                       # sigmoid: 0.5, 0.75
    gt = torch.tensor([[3.0, 4.0], [10.0, 13.0], [0.0, 0.0]])
    cost = m.point_cost(points, objectness, gt)
    expected = torch.tensor([
        [5 / 2 - 1.0, math.hypot(10, 13) / 2 - 1.0, 0 - 1.0],
        [math.hypot(7, 6) / 2 - 1.5, 3 / 2 - 1.5, math.hypot(10, 10) / 2 - 1.5],
    ])
    torch.testing.assert_close(cost, expected, rtol=1e-5, atol=1e-5)
    pi, gi = m.match_points(points, objectness, gt)
    assert dict(zip(pi.tolist(), gi.tolist())) == {0: 2, 1: 1}


# --- detr_criterion -------------------------------------------------------------------------------

def _detr_criterion(**overrides):
    return DetectionCriterion.create("detr_criterion", num_classes=2, **overrides)


def _big_logit(cls, K=2, value=6.0):
    out = torch.zeros(K + 1)
    out[cls] = value
    return out


def _single_image_case(pred_box_1):
    """3 queries, one target of class 1 at XYXY (10, 20, 30, 60) in a 100 x 50 (h x w) image,
    i.e. normalised cxcywh (0.4, 0.4, 0.4, 0.4). Query 1 is the obvious match."""
    logits = torch.stack([_big_logit(2), torch.tensor([0.5, 3.0, -1.0]), _big_logit(2)])[None]
    boxes = torch.tensor([[[0.9, 0.9, 0.05, 0.05], pred_box_1, [0.1, 0.9, 0.05, 0.05]]])
    raw = HeadOutput(logits=logits, boxes=boxes, extra={keys.IMAGE_SIZES: [(100, 50)]})
    targets = [{"boxes": torch.tensor([[10.0, 20.0, 30.0, 60.0]]), "labels": torch.tensor([1])}]
    return raw, targets


def _weighted_ce(logits, target_classes, eos_coef=0.1, K=2):
    w = torch.tensor([1.0] * K + [eos_coef])[target_classes]
    nll = -logits.log_softmax(-1).gather(-1, target_classes[:, None]).squeeze(-1)
    return (w * nll).sum() / w.sum()


def test_detr_criterion_is_registered():
    assert "detr_criterion" in DetectionCriterion.list_all()
    crit = _detr_criterion(matcher=ComponentRef("hungarian_matcher", {"cost_giou": 1.0}))
    assert crit.matcher.cfg.cost_giou == 1.0


def test_detr_criterion_perfect_box_hand_values():
    raw, targets = _single_image_case([0.4, 0.4, 0.4, 0.4])
    losses = _detr_criterion(aux_loss=False).compute_losses(raw, targets)
    assert set(losses) == {"loss_ce", "loss_bbox", "loss_giou"}
    expected_ce = _weighted_ce(raw.logits[0], torch.tensor([2, 1, 2]))
    torch.testing.assert_close(losses["loss_ce"], expected_ce)
    assert losses["loss_bbox"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["loss_giou"].item() == pytest.approx(0.0, abs=1e-6)


def test_detr_criterion_imperfect_box_hand_values_and_eos_coef():
    raw, targets = _single_image_case([0.45, 0.4, 0.4, 0.4])
    losses = _detr_criterion(aux_loss=False, eos_coef=0.2, loss_ce=2.0).compute_losses(raw, targets)
    # L1: 0.05 ; GIoU: pred XYXY (0.25, 0.2, 0.65, 0.6) vs (0.2, 0.2, 0.6, 0.6): inter 0.14, union 0.18,
    # enclosing 0.18 -> GIoU = 7 / 9
    assert losses["loss_bbox"].item() == pytest.approx(5 * 0.05, rel=1e-5)
    assert losses["loss_giou"].item() == pytest.approx(2 * (1 - 7 / 9), rel=1e-5)
    expected_ce = 2.0 * _weighted_ce(raw.logits[0], torch.tensor([2, 1, 2]), eos_coef=0.2)
    torch.testing.assert_close(losses["loss_ce"], expected_ce)


def test_detr_criterion_normalises_by_number_of_targets_in_the_batch():
    # image 0 (h 100, w 200): two targets, image 1 (h 50, w 50): one target. Queries 0..1 / 0 match.
    gt0 = torch.tensor([[0.25, 0.5, 0.1, 0.2], [0.75, 0.25, 0.2, 0.1]])
    gt1 = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    err = torch.tensor([[0.01, 0.0, 0.0, 0.0], [0.0, 0.02, 0.0, 0.0]])
    boxes = torch.stack([
        torch.cat([gt0 + err, torch.tensor([[0.05, 0.05, 0.02, 0.02]])]),
        torch.cat([gt1 + torch.tensor([[0.0, 0.0, 0.03, 0.0]]), torch.tensor([[0.95, 0.95, 0.02, 0.02]] * 2)]),
    ])
    logits = torch.stack([
        torch.stack([_big_logit(0), _big_logit(1), _big_logit(2)]),
        torch.stack([_big_logit(1), _big_logit(2), _big_logit(2)]),
    ])
    to_xyxy = lambda b, h, w: torchvision.ops.box_convert(b, "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
    targets = [
        {"boxes": to_xyxy(gt0, 100.0, 200.0), "labels": torch.tensor([0, 1])},
        {"boxes": to_xyxy(gt1, 50.0, 50.0), "labels": torch.tensor([1])},
    ]
    raw = HeadOutput(logits=logits, boxes=boxes, extra={keys.IMAGE_SIZES: [(100, 200), (50, 50)]})
    losses = _detr_criterion(aux_loss=False).compute_losses(raw, targets)
    assert losses["loss_bbox"].item() == pytest.approx(5 * (0.01 + 0.02 + 0.03) / 3, rel=1e-4)
    giou = torchvision.ops.generalized_box_iou
    pred = torch.cat([boxes[0, :2], boxes[1, :1]])
    true = torch.cat([gt0, gt1])
    expected_giou = (1 - torch.diag(giou(torchvision.ops.box_convert(pred, "cxcywh", "xyxy"),
                                         torchvision.ops.box_convert(true, "cxcywh", "xyxy")))).sum() / 3
    assert losses["loss_giou"].item() == pytest.approx(2 * expected_giou.item(), rel=1e-4)
    expected_ce = _weighted_ce(logits.reshape(6, 3), torch.tensor([0, 1, 2, 1, 2, 2]))
    torch.testing.assert_close(losses["loss_ce"], expected_ce)


def test_detr_criterion_aux_losses():
    raw, targets = _single_image_case([0.45, 0.4, 0.4, 0.4])
    aux0, _ = _single_image_case([0.4, 0.4, 0.4, 0.4])
    aux1, _ = _single_image_case([0.4, 0.45, 0.4, 0.4])
    raw.extra[keys.AUX_OUTPUTS] = [HeadOutput(logits=aux0.logits, boxes=aux0.boxes),
                                   HeadOutput(logits=aux1.logits, boxes=aux1.boxes)]
    crit = _detr_criterion()
    losses = crit.compute_losses(raw, targets)
    assert set(losses) == {f"loss_{n}{s}" for n in ("ce", "bbox", "giou") for s in ("", "_aux0", "_aux1")}
    alone = _detr_criterion(aux_loss=False)
    for i, aux in enumerate([aux0, aux1]):
        ref = alone.compute_losses(aux, targets)
        for name, value in ref.items():
            torch.testing.assert_close(losses[f"{name}_aux{i}"], value)
    assert set(_detr_criterion(aux_loss=False).compute_losses(raw, targets)) == {"loss_ce", "loss_bbox", "loss_giou"}


def test_detr_criterion_ignores_padded_queries():
    raw, targets = _single_image_case([0.45, 0.4, 0.4, 0.4])
    # add a padded query (index 0) with a perfect box and a confident correct class
    logits = torch.cat([_big_logit(1, value=10.0)[None, None], raw.logits], 1)
    boxes = torch.cat([torch.tensor([[[0.4, 0.4, 0.4, 0.4]]]), raw.boxes], 1)
    padded = HeadOutput(logits=logits, boxes=boxes, extra={
        keys.IMAGE_SIZES: [(100, 50)], keys.QUERY_MASK: torch.tensor([[False, True, True, True]])})
    crit = _detr_criterion(aux_loss=False)
    got, ref = crit.compute_losses(padded, targets), crit.compute_losses(raw, targets)
    for name in ref:
        torch.testing.assert_close(got[name], ref[name])
    assert got["loss_bbox"].item() > 0      # the padded perfect query was not matched


def test_detr_criterion_handles_images_without_targets():
    raw, _ = _single_image_case([0.45, 0.4, 0.4, 0.4])
    raw.logits.requires_grad_(True)
    raw.boxes.requires_grad_(True)
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    losses = _detr_criterion(aux_loss=False).compute_losses(raw, targets)
    expected_ce = F.cross_entropy(raw.logits[0], torch.tensor([2, 2, 2]))     # all "no object"
    torch.testing.assert_close(losses["loss_ce"], expected_ce)
    assert losses["loss_bbox"].item() == 0 and losses["loss_giou"].item() == 0
    sum(losses.values()).backward()
    assert torch.isfinite(raw.logits.grad).all()


def test_detr_criterion_validation_errors_and_metrics():
    raw, targets = _single_image_case([0.4, 0.4, 0.4, 0.4])
    crit = _detr_criterion()
    crit.validate_predictions(raw)
    with pytest.raises(ValueError, match="no object"):
        DetectionCriterion.create("detr_criterion", num_classes=3).validate_predictions(raw)
    with pytest.raises(ValueError, match="image_sizes"):
        crit.compute_losses(HeadOutput(logits=raw.logits, boxes=raw.boxes), targets)
    assert crit.compute_metrics(raw, targets) == {"matched_class_accuracy": 1.0}


# --- point_set_losses -------------------------------------------------------------------------------

def _point_case():
    points = torch.tensor([[[10.0, 10.0], [0.0, 0.0], [5.0, 5.0]]])
    objectness = torch.tensor([[0.0, 1.0, math.log(3.0)]])
    match = torch.tensor([[0, -1, 1]])
    gt = torch.tensor([[[12.0, 7.0], [5.0, 9.0]]])
    return points, objectness, match, gt


def _focal(x, positive, alpha=0.25, gamma=2.0):
    p = 1 / (1 + math.exp(-x))
    if positive:
        return alpha * (1 - p) ** gamma * -math.log(p)
    return (1 - alpha) * p ** gamma * -math.log(1 - p)


def test_point_set_losses_l1_hand_values():
    points, obj, match, gt = _point_case()
    loc, objl = point_set_losses(points, obj, match, torch.ones(1, 3, dtype=torch.bool), gt,
                                 loc_scale=2.0, loc_weight=5.0, objectness_weight=2.0, noobj_weight=0.5)
    # deltas (-2, 3) and (0, -4) px -> / 2 -> 1 + 1.5 + 0 + 2 = 4.5, over 2 matches
    assert loc.item() == pytest.approx(5.0 * 4.5 / 2, rel=1e-5)
    expected_obj = (_focal(0.0, True) + 0.5 * _focal(1.0, False) + _focal(math.log(3.0), True)) / 2
    assert objl.item() == pytest.approx(2.0 * expected_obj, rel=1e-5)


def test_point_set_losses_laplace_nll_hand_values():
    points, obj, match, gt = _point_case()
    log_scales = torch.log(torch.tensor([[[4.0, 4.0], [1.0, 1.0], [2.0, 2.0]]]))
    loc, _ = point_set_losses(points, obj, match, torch.ones(1, 3, dtype=torch.bool), gt,
                              log_scales=log_scales, loc_loss="laplace_nll", loc_scale=2.0, loc_weight=1.0)
    # in units of 2 px: q0 scale 2, |delta| (1, 1.5); q2 scale 1, |delta| (0, 2)
    expected = ((1 / 2 + math.log(2)) + (1.5 / 2 + math.log(2)) + (0 + 0) + (2 + 0)) / 2
    assert loc.item() == pytest.approx(expected, rel=1e-5)
    with pytest.raises(ValueError, match="log scales"):
        point_set_losses(points, obj, match, torch.ones(1, 3, dtype=torch.bool), gt, loc_loss="laplace_nll")
    with pytest.raises(ValueError, match="loc_loss"):
        point_set_losses(points, obj, match, torch.ones(1, 3, dtype=torch.bool), gt, loc_loss="l2")


def test_point_set_losses_num_targets_and_query_mask():
    points, obj, match, gt = _point_case()
    qmask = torch.tensor([[True, False, True]])                      # query 1 (a negative) is padding
    loc, objl = point_set_losses(points, obj, match, qmask, gt, loc_weight=1.0, objectness_weight=1.0,
                                 num_targets=4)
    assert loc.item() == pytest.approx((2 + 3 + 0 + 4) / 4, rel=1e-5)
    assert objl.item() == pytest.approx((_focal(0.0, True) + _focal(math.log(3.0), True)) / 4, rel=1e-5)
    # a padded query with a match index is neither a positive nor regressed
    loc2, _ = point_set_losses(points, obj, match, torch.tensor([[False, True, True]]), gt, loc_weight=1.0)
    assert loc2.item() == pytest.approx((0 + 4) / 1, rel=1e-5)


def test_point_set_losses_without_targets():
    points, obj, _, _ = _point_case()
    loc, objl = point_set_losses(points, obj, torch.full((1, 3), -1), torch.ones(1, 3, dtype=torch.bool),
                                 torch.zeros(1, 0, 2), objectness_weight=1.0)
    assert loc.item() == 0
    expected = _focal(0.0, False) + _focal(1.0, False) + _focal(math.log(3.0), False)   # normaliser clamps to 1
    assert objl.item() == pytest.approx(expected, rel=1e-5)


# --- point_set_criterion ---------------------------------------------------------------------------------

def test_point_set_criterion_registered_and_rejects_bad_loss():
    assert "point_set_criterion" in DetectionCriterion.list_all()
    with pytest.raises(ValueError, match="loc_loss"):
        DetectionCriterion.create("point_set_criterion", loc_loss="l2")


def test_point_set_criterion_converts_to_pixels_with_image_sizes():
    # image 0: h 100, w 200 ; image 1: h 40, w 80
    pts = torch.tensor([[[0.5, 0.5], [0.1, 0.9]], [[0.25, 0.75], [0.9, 0.1]]])
    scores = torch.tensor([[1.0, -1.0], [0.5, -2.0]])
    log_scales = torch.zeros(2, 2, 2)
    raw = HeadOutput(scores=scores, extra={keys.POINTS: pts, keys.POINT_LOG_SCALES: log_scales,
                                           keys.IMAGE_SIZES: [(100, 200), (40, 80)]})
    targets = [{"points": torch.tensor([[103.0, 46.0]])}, {"points": torch.tensor([[20.0, 32.0], [70.0, 5.0]])}]
    # pixels: image 0 q0 (100, 50) <- (103, 46); image 1 q0 (20, 30) <- (20, 32), q1 (72, 4) <- (70, 5)
    crit = DetectionCriterion.create("point_set_criterion", loc_scale=2.0, loc_weight=1.0, aux_loss=False)
    crit.validate_predictions(raw)
    losses = crit.compute_losses(raw, targets)
    assert set(losses) == {"loss_loc", "loss_objectness"}
    assert losses["loss_loc"].item() == pytest.approx((3 + 4 + 0 + 2 + 2 + 1) / 2 / 3, rel=1e-5)

    pixels = pts * torch.tensor([[[200.0, 100.0]], [[80.0, 40.0]]])
    gt = torch.tensor([[[103.0, 46.0], [0.0, 0.0]], [[20.0, 32.0], [70.0, 5.0]]])
    match = torch.tensor([[0, -1], [0, 1]])
    ref_loc, ref_obj = point_set_losses(pixels, scores, match, torch.ones(2, 2, dtype=torch.bool), gt,
                                        loc_scale=2.0, loc_weight=1.0, num_targets=3)
    torch.testing.assert_close(losses["loss_loc"], ref_loc)
    torch.testing.assert_close(losses["loss_objectness"], ref_obj)

    # Laplace: normalised log scales 0 -> pixel scales (w, h)
    lap = DetectionCriterion.create("point_set_criterion", loc_loss="laplace_nll", loc_weight=1.0, aux_loss=False)
    ref_lap, _ = point_set_losses(pixels, scores, match, torch.ones(2, 2, dtype=torch.bool), gt,
                                  log_scales=torch.log(torch.tensor([[[200.0, 100.0]] * 2, [[80.0, 40.0]] * 2])),
                                  loc_loss="laplace_nll", loc_weight=1.0, num_targets=3)
    torch.testing.assert_close(lap.compute_losses(raw, targets)["loss_loc"], ref_lap)

    assert crit.compute_metrics(raw, targets)["loc_error_px"] == pytest.approx(
        (5 + 2 + math.hypot(2, 1)) / 3, rel=1e-5)


def test_point_set_criterion_aux_losses_and_validation():
    pts = torch.tensor([[[0.5, 0.5]]])
    raw = HeadOutput(scores=torch.zeros(1, 1), extra={
        keys.POINTS: pts, keys.IMAGE_SIZES: [(10, 10)],
        keys.AUX_OUTPUTS: [HeadOutput(scores=torch.zeros(1, 1), extra={keys.POINTS: pts + 0.1})]})
    targets = [{"points": torch.tensor([[5.0, 5.0]])}]
    losses = DetectionCriterion.create("point_set_criterion", loc_weight=1.0).compute_losses(raw, targets)
    assert set(losses) == {"loss_loc", "loss_objectness", "loss_loc_aux0", "loss_objectness_aux0"}
    assert losses["loss_loc"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["loss_loc_aux0"].item() == pytest.approx(2.0, rel=1e-5)
    with pytest.raises(ValueError, match="points"):
        DetectionCriterion.create("point_set_criterion").validate_predictions(HeadOutput(scores=torch.zeros(1, 1)))
    with pytest.raises(ValueError, match="predict_uncertainty"):
        DetectionCriterion.create("point_set_criterion", loc_loss="laplace_nll").validate_predictions(raw)


def test_point_set_criterion_empty_targets_and_padded_queries():
    pts = torch.tensor([[[0.5, 0.5], [0.2, 0.2]]])
    scores = torch.tensor([[0.0, 2.0]])
    crit = DetectionCriterion.create("point_set_criterion", objectness_weight=1.0, aux_loss=False)
    raw = HeadOutput(scores=scores, extra={keys.POINTS: pts, keys.IMAGE_SIZES: [(10, 10)],
                                           keys.QUERY_MASK: torch.tensor([[True, False]])})
    losses = crit.compute_losses(raw, [{"points": torch.zeros(0, 2)}])
    assert losses["loss_loc"].item() == 0
    assert losses["loss_objectness"].item() == pytest.approx(_focal(0.0, False), rel=1e-5)
    # the padded query sits exactly on the target but must not be matched
    losses = crit.compute_losses(raw, [{"points": torch.tensor([[2.0, 2.0]])}])
    assert losses["loss_loc"].item() == pytest.approx(5.0 * 6.0, rel=1e-5)


# --- postprocessors ---------------------------------------------------------------------------------

def test_detr_postprocessor_converts_boxes_and_respects_thresh_topk_and_mask():
    # K = 2; image h 100, w 200
    logits = torch.log(torch.tensor([[
        [0.7, 0.2, 0.1],     # class 0 @ 0.7
        [0.1, 0.6, 0.3],     # class 1 @ 0.6
        [0.05, 0.05, 0.9],   # best foreground 0.05 -> below thresh
        [0.9, 0.05, 0.05],   # padded query
        [0.3, 0.5, 0.2],     # class 1 @ 0.5
    ]]))
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.95, 0.1, 0.2, 0.4], [0.5, 0.5, 0.1, 0.1],
                           [0.5, 0.5, 0.1, 0.1], [0.25, 0.25, 0.1, 0.1]]])
    raw = HeadOutput(logits=logits, boxes=boxes, extra={
        keys.IMAGE_SIZES: [(100, 200)], keys.QUERY_MASK: torch.tensor([[True, True, True, False, True]])})
    assert "detr_postprocessor" in Postprocessor.list_all()
    (s,) = Postprocessor.create("detr_postprocessor", score_thresh=0.1).process(raw, num_classes=2)
    torch.testing.assert_close(s.target["scores"], torch.tensor([0.7, 0.6, 0.5]))
    assert s.target["labels"].tolist() == [0, 1, 1]
    torch.testing.assert_close(s.target["boxes"], torch.tensor([
        [80.0, 30.0, 120.0, 70.0],
        [170.0, 0.0, 200.0, 30.0],       # clipped from (170, -10, 210, 30)
        [40.0, 20.0, 60.0, 30.0],
    ]))
    assert s.meta == {"image_index": 0, "image_size": (100, 200)}
    (top,) = Postprocessor.create("detr_postprocessor", detections_per_image=2).process(raw, num_classes=2)
    torch.testing.assert_close(top.target["scores"], torch.tensor([0.7, 0.6]))


def test_point_postprocessor_converts_to_pixels_and_respects_thresh_topk_and_mask():
    scores = torch.logit(torch.tensor([[0.9, 0.3, 0.6, 0.99], [0.8, 0.7, 0.1, 0.1]]))
    pts = torch.tensor([[[0.5, 0.5], [0.1, 0.1], [0.25, 0.75], [0.0, 0.0]],
                        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.5], [0.5, 0.5]]])
    raw = HeadOutput(scores=scores, extra={
        keys.POINTS: pts, keys.IMAGE_SIZES: [(100, 200), (10, 20)],
        keys.QUERY_MASK: torch.tensor([[True, True, True, False], [True, True, True, True]])})
    assert "point_postprocessor" in Postprocessor.list_all()
    a, b = Postprocessor.create("point_postprocessor").process(raw)
    torch.testing.assert_close(a.target["scores"], torch.tensor([0.9, 0.6]))
    torch.testing.assert_close(a.target["points"], torch.tensor([[100.0, 50.0], [50.0, 75.0]]))
    torch.testing.assert_close(b.target["points"], torch.tensor([[2.0, 2.0], [6.0, 4.0]]))
    assert b.meta["image_size"] == (10, 20)
    a, b = Postprocessor.create("point_postprocessor", score_thresh=0.0, max_points=1).process(raw)
    torch.testing.assert_close(a.target["scores"], torch.tensor([0.9]))
    assert b.target["points"].shape == (1, 2)


# --- batched matching (one host transfer per layer) -----------------------------------------

from optastra.detection.matching.hungarian import batched_generalized_box_iou, batched_linear_assignment


def _per_image_match(cost, mask, counts):
    out = torch.full(mask.shape, -1, dtype=torch.long)
    for b, n in enumerate(counts):
        rows = torch.nonzero(mask[b]).flatten()
        r, c = linear_assignment(cost[b][rows][:, :n])
        out[b, rows[r]] = c
    return out


def test_batched_linear_assignment_equals_per_image():
    torch.manual_seed(0)
    B, M, G = 6, 9, 5
    cost = torch.rand(B, M, G)
    mask = torch.rand(B, M) > 0.25
    counts = [5, 0, 3, 1, 5, 2]
    assert torch.equal(batched_linear_assignment(cost, mask, counts), _per_image_match(cost, mask, counts))


def test_batched_linear_assignment_ignores_padding_and_masked_rows():
    cost = torch.tensor([[[0.0, -100.0], [5.0, -100.0]]])     # column 1 is padding (count 1): must not be used
    mask = torch.tensor([[False, True]])                      # row 0 is a padded query: must not be used
    assert batched_linear_assignment(cost, mask, [1]).tolist() == [[-1, 0]]


def test_batched_giou_and_costs_equal_per_image():
    torch.manual_seed(0)
    m = HungarianMatcher()
    B, M, G, K = 3, 7, 4, 5
    cx = torch.rand(B, M, 2) * 0.6 + 0.2
    boxes = torch.cat([cx, torch.rand(B, M, 2) * 0.3 + 0.05], -1)
    gt_boxes = torch.cat([torch.rand(B, G, 2) * 0.6 + 0.2, torch.rand(B, G, 2) * 0.3 + 0.05], -1)
    xyxy = lambda b: torchvision.ops.box_convert(b, "cxcywh", "xyxy")
    giou = batched_generalized_box_iou(xyxy(boxes), xyxy(gt_boxes))
    for b in range(B):
        torch.testing.assert_close(giou[b], torchvision.ops.generalized_box_iou(xyxy(boxes[b]), xyxy(gt_boxes[b])))
    logits, labels = torch.randn(B, M, K + 1), torch.randint(0, K, (B, G))
    batched = m.box_cost_batched(logits, boxes, labels, gt_boxes)
    points, obj, gt_pts = torch.rand(B, M, 2) * 64, torch.randn(B, M), torch.rand(B, G, 2) * 64
    pbatched = m.point_cost_batched(points, obj, gt_pts)
    for b in range(B):
        torch.testing.assert_close(batched[b], m.box_cost(logits[b], boxes[b], labels[b], gt_boxes[b]))
        torch.testing.assert_close(pbatched[b], m.point_cost(points[b], obj[b], gt_pts[b]))
