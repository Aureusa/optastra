import pytest
import torch
import torch.nn as nn

from optastra.optim import Optimizer, ParamGroupConfig

# Import for registry side effects.
import optastra.optim.adam  # noqa: F401
import optastra.optim.adamw  # noqa: F401
import optastra.optim.sgd  # noqa: F401


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.head = nn.Linear(4, 2)


class _ToyViT(nn.Module):
    """Mimics ViT parameter naming: a bare `cls_token` and a nested `pos_embed.pos_embed`."""

    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        self.pos_embed = nn.Module()
        self.pos_embed.pos_embed = nn.Parameter(torch.zeros(1, 5, 4))
        self.proj = nn.Linear(4, 4)


def _wd_by_param(optimizer: torch.optim.Optimizer) -> dict[int, float]:
    return {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}


def test_optimizer_create_builds_adam_with_overrides_and_param_groups():
    model = _ToyModel()
    optimizer = Optimizer.create(
        "adam",
        model,
        lr=0.005,
        param_groups=ParamGroupConfig(lr_multipliers={"backbone": 0.1}),
    )

    assert isinstance(optimizer, torch.optim.Adam)
    lrs = sorted({group["lr"] for group in optimizer.param_groups})
    assert lrs == [0.0005, 0.005]


def test_optimizer_create_builds_adamw_and_sgd():
    model = _ToyModel()

    adamw = Optimizer.create("adamw", model, lr=0.001)
    sgd = Optimizer.create("sgd", model, lr=0.1, momentum=0.8)

    assert isinstance(adamw, torch.optim.AdamW)
    assert isinstance(sgd, torch.optim.SGD)
    assert adamw.defaults["weight_decay"] == 0.01
    assert sgd.defaults["momentum"] == 0.8


@pytest.mark.parametrize("name, weight_decay", [("adam", 0.02), ("adamw", 0.05), ("sgd", 1e-3)])
def test_optimizer_param_groups_carry_configured_weight_decay(name, weight_decay):
    # Regression: build_param_groups used to be called without base_weight_decay,
    # so every group had weight_decay=0.0 regardless of the config.
    model = _ToyModel()
    optimizer = Optimizer.create(name, model, weight_decay=weight_decay)
    wd = _wd_by_param(optimizer)

    assert wd[id(model.backbone[0].weight)] == weight_decay
    assert wd[id(model.head.weight)] == weight_decay
    assert wd[id(model.backbone[0].bias)] == 0.0
    assert wd[id(model.head.bias)] == 0.0
    assert wd[id(model.backbone[1].weight)] == 0.0   # BatchNorm weight
    assert wd[id(model.backbone[1].bias)] == 0.0


def test_optimizer_default_weight_decay_reaches_param_groups():
    model = _ToyModel()
    optimizer = Optimizer.create("adamw", model)
    assert _wd_by_param(optimizer)[id(model.head.weight)] == 0.01


def test_optimizer_excludes_pos_embed_and_cls_token_from_weight_decay():
    model = _ToyViT()
    optimizer = Optimizer.create("adamw", model, weight_decay=0.05)
    wd = _wd_by_param(optimizer)

    assert wd[id(model.cls_token)] == 0.0
    assert wd[id(model.pos_embed.pos_embed)] == 0.0
    assert wd[id(model.proj.weight)] == 0.05


def test_decay_actually_applied_by_adamw_step():
    # End-to-end: with zero gradients AdamW's only effect is decoupled decay,
    # w <- w * (1 - lr * wd), which used to be a no-op because wd was 0.
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = Optimizer.create("adamw", model, lr=0.1, weight_decay=0.5)
    model.weight.grad = torch.zeros_like(model.weight)
    optimizer.step()
    assert torch.allclose(model.weight, torch.full((2, 2), 1.0 - 0.1 * 0.5))


def test_optimizer_create_rejects_unknown_name():
    model = _ToyModel()

    try:
        Optimizer.create("missing_optimizer", model)
        assert False, "Expected ValueError for missing optimizer"
    except ValueError as e:
        assert "not registered" in str(e)
