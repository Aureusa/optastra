import torch.nn as nn

from optastra.optim.param_groups import ParamGroupConfig, build_param_groups


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
        )
        self.head = nn.Sequential(
            nn.Linear(4, 2),
            nn.LayerNorm(2),
        )


def test_build_param_groups_splits_bias_and_norm_parameters():
    model = _ToyModel()
    groups = build_param_groups(model, ParamGroupConfig(no_decay_norm_and_bias=True), base_lr=0.01, base_weight_decay=0.1)

    assert groups
    no_decay_groups = [group for group in groups if group["weight_decay"] == 0.0]
    decay_groups = [group for group in groups if group["weight_decay"] == 0.1]

    assert no_decay_groups
    assert decay_groups


def test_build_param_groups_applies_longest_lr_prefix_match():
    model = _ToyModel()
    cfg = ParamGroupConfig(lr_multipliers={"backbone": 0.1, "backbone.0": 0.01, "head": 0.5})

    groups = build_param_groups(model, cfg, base_lr=0.02, base_weight_decay=0.0)
    lrs = sorted({group["lr"] for group in groups})

    assert 0.0002 in lrs  # backbone.0 -> 0.01 x 0.02
    assert 0.002 in lrs   # backbone -> 0.1 x 0.02
    assert 0.01 in lrs    # head -> 0.5 x 0.02
