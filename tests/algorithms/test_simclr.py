import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from optastra import Algorithm
from optastra.algorithms.simclr.model import SimCLRModel
from optastra.algorithms.simclr.routine import SimCLRConfig, SimCLRTask, nt_xent_loss

SMALL_MLP = {"hidden_dim": 32, "out_dim": 16}


class _ToySimCLRModel(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(3 * 8 * 8, embed_dim)

    def forward(self, views: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.proj(v.flatten(1)) for v in views]


def _reference_nt_xent(views: list[torch.Tensor], temperature: float) -> float:
    """Straight-from-the-paper loops: every (view, image) pair is an anchor,
    positives are the other views of the same image, the denominator runs
    over every other embedding in the batch."""
    z = [F.normalize(v.double(), dim=-1) for v in views]
    num_views, n = len(z), z[0].shape[0]
    anchors = [(a, i) for a in range(num_views) for i in range(n)]

    def sim(x, y):
        return float(z[x[0]][x[1]] @ z[y[0]][y[1]]) / temperature

    total = 0.0
    for anchor in anchors:
        denom = sum(math.exp(sim(anchor, other)) for other in anchors if other != anchor)
        positives = [(b, anchor[1]) for b in range(num_views) if b != anchor[0]]
        total += -sum(math.log(math.exp(sim(anchor, p)) / denom) for p in positives) / len(positives)
    return total / len(anchors)


@pytest.mark.parametrize("num_views", [2, 3])
@pytest.mark.parametrize("temperature", [0.1, 0.5])
def test_nt_xent_matches_reference(num_views, temperature):
    torch.manual_seed(0)
    views = [torch.randn(5, 8) for _ in range(num_views)]
    expected = _reference_nt_xent(views, temperature)
    assert nt_xent_loss(views, temperature).item() == pytest.approx(expected, rel=1e-5)


def test_nt_xent_is_symmetric_in_views_and_batch_order():
    torch.manual_seed(1)
    z1, z2, z3 = torch.randn(6, 8), torch.randn(6, 8), torch.randn(6, 8)
    perm = torch.randperm(6)

    loss = nt_xent_loss([z1, z2], 0.2)
    assert nt_xent_loss([z2, z1], 0.2).item() == pytest.approx(loss.item(), rel=1e-6)
    assert nt_xent_loss([z1[perm], z2[perm]], 0.2).item() == pytest.approx(loss.item(), rel=1e-6)

    loss3 = nt_xent_loss([z1, z2, z3], 0.2)
    assert nt_xent_loss([z3, z1, z2], 0.2).item() == pytest.approx(loss3.item(), rel=1e-6)


def test_nt_xent_uses_intra_view_negatives():
    """Orthogonal images, identical views: each anchor has one positive
    (cos 1) and 2N-2 negatives (cos 0) -- N-1 from the other view AND N-1
    from its own view. A cross-view-only loss would count just N-1."""
    n, temperature = 4, 0.5
    z = torch.eye(n)
    expected = -math.log(math.exp(1 / temperature) / (math.exp(1 / temperature) + 2 * n - 2))
    assert nt_xent_loss([z, z], temperature).item() == pytest.approx(expected, rel=1e-6)


def test_simclr_task_train_step_returns_nt_xent_loss():
    torch.manual_seed(0)
    model = _ToySimCLRModel()
    task = SimCLRTask(SimCLRConfig(temperature=0.5))
    views = [torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)]

    output = task.run_step(model, {"views": views}, stage="train")

    expected = _reference_nt_xent([e.detach() for e in model(views)], 0.5)
    assert output.losses["nt_xent_loss"].item() == pytest.approx(expected, rel=1e-5)
    assert output.loss is output.losses["nt_xent_loss"]


def test_simclr_task_supports_more_than_two_views():
    model = _ToySimCLRModel()
    task = SimCLRTask(SimCLRConfig())
    views = [torch.randn(4, 3, 8, 8) for _ in range(4)]
    output = task.run_step(model, {"views": views}, stage="train")
    expected = _reference_nt_xent([e.detach() for e in model(views)], 0.5)
    assert output.loss.item() == pytest.approx(expected, rel=1e-5)


def test_simclr_task_predict_step_decodes_raw_predictions():
    model = _ToySimCLRModel()
    task = SimCLRTask(SimCLRConfig())
    views = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)]

    output = task.run_step(model, {"views": views}, stage="predict")

    assert output.loss is None
    assert isinstance(output.predictions, list)
    assert len(output.predictions) == 2
    assert output.predictions[0].shape[0] == 2


def test_simclr_task_requires_minimum_number_of_views():
    model = _ToySimCLRModel()
    task = SimCLRTask(SimCLRConfig())

    with pytest.raises(ValueError, match="requires >= 2 views"):
        task.run_step(model, {"views": [torch.randn(2, 3, 8, 8)]}, stage="train")


def test_simclr_builds_model_from_config_with_cnn_backbone():
    algo = Algorithm.create("simclr", backbone="resnet18", projector=SMALL_MLP)
    model = algo.build_model()

    assert isinstance(model, SimCLRModel)
    assert type(model.backbone).__name__ == "ResNet"
    assert type(model.neck).__name__ == "GlobalPool"   # CNN -> default pooling neck
    out = model([torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)])
    assert [o.shape for o in out] == [torch.Size([2, 16])] * 2


def test_simclr_swaps_to_a_vit_backbone_by_config_only():
    algo = Algorithm.create(
        "simclr",
        backbone=("vit_tiny", {"img_size": 32, "patch_size": 8, "depth": 1}),
        projector=SMALL_MLP,
    )
    model = algo.build_model()

    assert type(model.backbone).__name__ == "ViT"
    assert model.neck is None   # ViT already pools (CLS token)
    assert model.projector.mlp[0].in_features == 192
    loss = algo.run_step(model, {"views": [torch.randn(3, 3, 32, 32)] * 2}).loss
    assert torch.isfinite(loss)


def test_simclr_model_without_pooled_output_fails_clearly():
    from optastra import Backbone
    backbone = Backbone.create("resnet18")
    model = SimCLRModel(backbone, None, nn.Identity())
    with pytest.raises(ValueError, match="pooling neck"):
        model([torch.randn(2, 3, 32, 32)])
