import torch
import torch.nn as nn

from optastra import Optimizer
from optastra.optim import LARS, LARSConfig


def test_lars_is_registered():
    assert "lars" in Optimizer.list_all()
    assert isinstance(Optimizer.create("lars", nn.Linear(2, 2)), LARS)


def test_lars_matches_hand_computed_update():
    w = nn.Parameter(torch.tensor([3.0, 4.0]))          # ||w|| = 5
    w.grad = torch.tensor([0.6, 0.8])                    # ||g|| = 1
    cfg = LARSConfig(lr=0.1, momentum=0.9, weight_decay=0.0, eta=0.01, exclude_zero_decay=False)
    opt = LARS([{"params": [w], "weight_decay": 0.0}], cfg)
    opt.step()
    # trust = eta * ||w|| / ||g|| = 0.05 -> update = 0.05 * g; w -= lr * update
    assert torch.allclose(w.detach(), torch.tensor([3.0, 4.0]) - 0.1 * 0.05 * torch.tensor([0.6, 0.8]))
    w.grad = torch.tensor([0.6, 0.8])
    w_before = w.detach().clone()
    opt.step()   # momentum buffer: 0.9 * old + new
    trust = 0.01 * w_before.norm() / w.grad.norm()
    expected = w_before - 0.1 * (0.9 * 0.05 * torch.tensor([0.6, 0.8]) + trust * torch.tensor([0.6, 0.8]))
    assert torch.allclose(w.detach(), expected, atol=1e-7)


def test_lars_weight_decay_enters_the_trust_ratio():
    w = nn.Parameter(torch.tensor([3.0, 4.0]))
    w.grad = torch.tensor([0.0, 1.0])
    opt = LARS([{"params": [w], "weight_decay": 0.1}], LARSConfig(lr=1.0, momentum=0.0, eta=0.01))
    u = torch.tensor([0.0, 1.0]) + 0.1 * torch.tensor([3.0, 4.0])
    expected = torch.tensor([3.0, 4.0]) - 0.01 * 5.0 / u.norm() * u
    opt.step()
    assert torch.allclose(w.detach(), expected)


def test_lars_does_not_rescale_zero_weights():
    w = nn.Parameter(torch.zeros(3))                     # e.g. a zero-initialized layer
    w.grad = torch.ones(3)
    opt = LARS([{"params": [w], "weight_decay": 0.0}], LARSConfig(lr=0.5, momentum=0.0, exclude_zero_decay=False))
    opt.step()
    assert torch.allclose(w.detach(), torch.full((3,), -0.5))   # plain SGD step, no division by ||w|| = 0


def test_lars_skips_adaptation_and_decay_for_norm_and_bias():
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    opt = Optimizer.create("lars", model, lr=0.5, weight_decay=1e-3)
    by_decay = {g["weight_decay"]: g for g in opt.param_groups}
    assert by_decay[1e-3]["lars_adapt"] and not by_decay[0.0]["lars_adapt"]
    decayed = {id(p) for p in by_decay[1e-3]["params"]}
    assert decayed == {id(model[0].weight)}   # linear bias + BN weight/bias are in the no-decay group
    # A zero-decay parameter moves by plain momentum SGD.
    bias = model[0].bias
    bias.grad = torch.ones_like(bias)
    for p in model.parameters():
        if p is not bias:
            p.grad = torch.zeros_like(p)
    before = bias.detach().clone()
    opt.step()
    assert torch.allclose(bias.detach(), before - 0.5)


def test_sync_batchnorm_parameters_count_as_norm_parameters():
    model = nn.SyncBatchNorm.convert_sync_batchnorm(nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))
    assert isinstance(model[1], nn.SyncBatchNorm)
    opt = Optimizer.create("lars", model, weight_decay=1e-3)
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert decayed == {id(model[0].weight)}
