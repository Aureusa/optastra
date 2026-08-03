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


def test_optimizer_create_rejects_unknown_name():
    model = _ToyModel()

    try:
        Optimizer.create("missing_optimizer", model)
        assert False, "Expected ValueError for missing optimizer"
    except ValueError as e:
        assert "not registered" in str(e)
