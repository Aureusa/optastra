import math

import pytest
import torch

from optastra.core.component_ref import ComponentRef
from optastra.detection import DetectionCriterion, keys
from optastra.detection.sampling.balanced_sampler import BalancedSampler, BalancedSamplerConfig
from optastra.nn.blocks.geometry.boxes import encode_boxes
from optastra.nn.features import FeatureMaps
from tests.proposal_generators._fixed_rpn import anchor_box, make_fixed_rpn


def _criterion(**overrides):
    return DetectionCriterion.create("rcnn_criterion", num_classes=3, **overrides)


def _rpn_output(rpn, sizes: dict[str, tuple[int, int]], batch: int = 1) -> FeatureMaps:
    feats = {name: torch.zeros(batch, 4, h, w) for name, (h, w) in sizes.items()}
    return rpn(FeatureMaps(feature_maps=feats))


def test_rpn_loss_pairs_each_objectness_logit_with_its_own_anchor():
    # One anchor (ratio 2.0 at cell y=1, x=3) is the GT box; it alone has a high score
    # and zero deltas. Every other anchor is confidently background with large deltas.
    objectness = torch.full((1, 3, 4, 4), -10.0)
    objectness[0, 2, 1, 3] = 10.0
    deltas = torch.full((1, 12, 4, 4), 3.0)
    deltas[0, 8:12, 1, 3] = 0.0
    rpn = make_fixed_rpn({"P3": objectness}, {"P3": deltas}, {"P3": 8})
    rpn_out = _rpn_output(rpn, {"P3": (4, 4)})
    gt = torch.tensor([anchor_box(28.0, 12.0, 8, 2.0)])

    losses = _criterion()._rpn_losses(rpn_out, [{"boxes": gt, "labels": torch.tensor([0])}])

    assert losses["rpn_objectness_loss"].item() < 1e-3
    assert losses["rpn_box_loss"].item() == pytest.approx(0.0, abs=1e-6)


def test_rpn_loss_encodes_targets_with_the_rpns_own_box_coder_weights():
    weights = (10.0, 10.0, 5.0, 5.0)
    anchor = torch.tensor([anchor_box(28.0, 12.0, 8, 2.0)])
    gt = anchor + torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # still clearly best-matched to that anchor
    objectness = torch.full((1, 3, 4, 4), -10.0)
    objectness[0, 2, 1, 3] = 10.0
    deltas = torch.zeros((1, 12, 4, 4))
    deltas[0, 8:12, 1, 3] = encode_boxes(anchor, gt, weights=weights)[0]
    rpn = make_fixed_rpn({"P3": objectness}, {"P3": deltas}, {"P3": 8}, bbox_reg_weights=weights)
    rpn_out = _rpn_output(rpn, {"P3": (4, 4)})

    # The ROI box coder (bbox_reg_weights of the criterion) is deliberately different.
    losses = _criterion(bbox_reg_weights=(1.0, 1.0, 1.0, 1.0))._rpn_losses(
        rpn_out, [{"boxes": gt, "labels": torch.tensor([0])}]
    )

    assert rpn_out.extra[keys.BOX_CODER_WEIGHTS] == weights
    assert losses["rpn_box_loss"].item() == pytest.approx(0.0, abs=1e-6)


class CountingSampler(BalancedSampler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls: list[int] = []

    def sample(self, labels, **kwargs):
        self.calls.append(labels.numel())
        return super().sample(labels, **kwargs)


def test_rpn_loss_matches_and_samples_all_levels_together_per_image():
    sizes = {"P3": (8, 8), "P4": (4, 4)}
    objectness = {name: torch.zeros((2, 3, h, w)) for name, (h, w) in sizes.items()}
    deltas = {name: torch.zeros((2, 12, h, w)) for name, (h, w) in sizes.items()}
    rpn = make_fixed_rpn(objectness, deltas, {"P3": 8, "P4": 16})
    rpn_out = _rpn_output(rpn, sizes, batch=2)
    num_anchors = 3 * (8 * 8 + 4 * 4)

    criterion = _criterion()
    criterion.rpn_sampler = CountingSampler(BalancedSamplerConfig(batch_size=32, positive_fraction=0.5))
    targets = [
        {"boxes": torch.tensor([[8.0, 8.0, 40.0, 40.0]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[20.0, 4.0, 60.0, 30.0], [2.0, 2.0, 12.0, 12.0]]), "labels": torch.tensor([1, 2])},
    ]
    losses = criterion._rpn_losses(rpn_out, targets)

    # One matching/sampling pass per image over the anchors of *all* levels ...
    assert criterion.rpn_sampler.calls == [num_anchors, num_anchors]
    # ... and one normalisation: with every logit 0, BCE of each sampled anchor is log(2),
    # so the mean is log(2) no matter how many levels there are (per-level sums gave 2 * log 2).
    assert losses["rpn_objectness_loss"].item() == pytest.approx(math.log(2.0), rel=1e-5)


def test_rpn_loss_without_gt_boxes_uses_only_negatives():
    objectness = torch.full((1, 3, 4, 4), -10.0)
    rpn = make_fixed_rpn({"P3": objectness}, {"P3": torch.ones((1, 12, 4, 4))}, {"P3": 8})
    rpn_out = _rpn_output(rpn, {"P3": (4, 4)})

    losses = _criterion()._rpn_losses(rpn_out, [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)}])

    assert losses["rpn_objectness_loss"].item() < 1e-3
    assert losses["rpn_box_loss"].item() == 0.0
