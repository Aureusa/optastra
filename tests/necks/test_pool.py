import pytest
import torch

from optastra.necks import Neck
from optastra.nn.features import FeatureMaps, FeatureSpec


def _cnn_spec():
    return FeatureSpec(channels={"C4": 8, "C5": 16}, strides={"C4": 16, "C5": 32})


def test_global_avg_pool_defaults_to_deepest_stage_and_averages():
    neck = Neck.create("global_avg_pool", _cnn_spec())
    assert neck.stage == "C5" and neck.out_spec.embed_dim == 16

    c5 = torch.randn(2, 16, 3, 3)
    out = neck(FeatureMaps(feature_maps={"C4": torch.randn(2, 8, 6, 6), "C5": c5}))
    torch.testing.assert_close(out.pooled, c5.mean(dim=(2, 3)))


def test_pool_necks_can_select_a_stage_and_reject_unknown_ones():
    neck = Neck.create("global_max_pool", _cnn_spec(), stage="C4")
    c4 = torch.randn(2, 8, 6, 6)
    out = neck(FeatureMaps(feature_maps={"C4": c4, "C5": torch.randn(2, 16, 3, 3)}))
    torch.testing.assert_close(out.pooled, c4.amax(dim=(2, 3)))

    with pytest.raises(ValueError, match="not found"):
        Neck.create("gem_pool", _cnn_spec(), stage="C9")


def test_gem_pool_with_p1_is_average_pool():
    neck = Neck.create("gem_pool", _cnn_spec(), p=1.0)
    c5 = torch.rand(2, 16, 3, 3) + 0.1
    out = neck(FeatureMaps(feature_maps={"C5": c5}))
    torch.testing.assert_close(out.pooled, c5.mean(dim=(2, 3)))


def test_token_pool_pools_patch_tokens_over_tokens():
    spec = FeatureSpec(embed_dim=6, num_tokens=4)
    tokens = torch.randn(2, 4, 6)

    mean_pool = Neck.create("token_pool", spec)
    torch.testing.assert_close(mean_pool(FeatureMaps(patch_tokens=tokens)).pooled, tokens.mean(dim=1))
    assert mean_pool.out_spec.embed_dim == 6

    max_pool = Neck.create("token_pool", spec, method="max")
    torch.testing.assert_close(max_pool(FeatureMaps(patch_tokens=tokens)).pooled, tokens.amax(dim=1))

    with pytest.raises(ValueError, match="patch_tokens"):
        mean_pool(FeatureMaps(feature_maps={"C4": torch.randn(2, 6, 2, 2)}))
