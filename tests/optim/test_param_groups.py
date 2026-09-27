import torch
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


def _wd_by_param(groups: list[dict]) -> dict[int, float]:
    return {id(p): group["weight_decay"] for group in groups for p in group["params"]}


def test_build_param_groups_splits_bias_and_norm_parameters():
    model = _ToyModel()
    groups = build_param_groups(model, ParamGroupConfig(no_decay_norm_and_bias=True), base_lr=0.01, base_weight_decay=0.1)
    wd = _wd_by_param(groups)

    assert wd[id(model.backbone[0].weight)] == 0.1
    assert wd[id(model.head[0].weight)] == 0.1
    for no_decay in (model.backbone[0].bias, model.backbone[1].weight, model.backbone[1].bias,
                     model.head[0].bias, model.head[1].weight, model.head[1].bias):
        assert wd[id(no_decay)] == 0.0


def test_build_param_groups_decays_everything_when_split_disabled():
    model = _ToyModel()
    groups = build_param_groups(model, ParamGroupConfig(no_decay_norm_and_bias=False), base_lr=0.01, base_weight_decay=0.1)
    assert set(_wd_by_param(groups).values()) == {0.1}


def test_build_param_groups_excludes_configured_names_from_decay():
    model = nn.Module()
    model.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
    model.pos_embed = nn.Module()
    model.pos_embed.pos_embed = nn.Parameter(torch.zeros(1, 4, 4))
    model.proj = nn.Linear(4, 4)

    wd = _wd_by_param(build_param_groups(model, ParamGroupConfig(), base_lr=0.01, base_weight_decay=0.1))
    assert wd[id(model.cls_token)] == 0.0
    assert wd[id(model.pos_embed.pos_embed)] == 0.0
    assert wd[id(model.proj.weight)] == 0.1

    wd = _wd_by_param(build_param_groups(
        model, ParamGroupConfig(no_decay_names=()), base_lr=0.01, base_weight_decay=0.1,
    ))
    assert wd[id(model.cls_token)] == 0.1


def test_build_param_groups_applies_longest_lr_prefix_match():
    model = _ToyModel()
    cfg = ParamGroupConfig(lr_multipliers={"backbone": 0.1, "backbone.0": 0.01, "head": 0.5})

    groups = build_param_groups(model, cfg, base_lr=0.02, base_weight_decay=0.0)
    lr = {id(p): group["lr"] for group in groups for p in group["params"]}

    assert lr[id(model.backbone[0].weight)] == 0.02 * 0.01   # backbone.0 wins over backbone
    assert lr[id(model.backbone[1].weight)] == 0.02 * 0.1
    assert lr[id(model.head[0].weight)] == 0.02 * 0.5
