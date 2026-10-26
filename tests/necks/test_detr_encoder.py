import pytest
import torch
import torch.nn as nn

from optastra.necks import DETREncoder, Neck
from optastra.nn.features import FeatureMaps, FeatureSpec


def _spec() -> FeatureSpec:
    return FeatureSpec(channels={"C3": 6, "C4": 10, "C5": 12}, strides={"C3": 8, "C4": 16, "C5": 32})


def _encoder(**overrides) -> DETREncoder:
    torch.manual_seed(0)
    kw = dict(d_model=16, num_layers=2, num_heads=2, dim_feedforward=32, dropout=0.0)
    kw.update(overrides)
    return Neck.create("detr_encoder", _spec(), **kw).eval()


def _maps(B, h4, w4, h5, w5, h3=None, w3=None):
    out = {"C4": torch.randn(B, 10, h4, w4), "C5": torch.randn(B, 12, h5, w5)}
    if h3 is not None:
        out["C3"] = torch.randn(B, 6, h3, w3)
    return out


def test_detr_encoder_is_registered():
    assert "detr_encoder" in Neck.list_all()
    cfg = Neck.get_default_config("detr_encoder")
    assert cfg.d_model == 256 and cfg.normalize_before is False and cfg.in_features == ()


def test_default_uses_only_the_coarsest_map():
    enc = _encoder()
    assert enc.feature_names == ["C5"]
    assert enc.level_embed is None
    assert isinstance(enc.norm, nn.Identity)                     # post-norm: no final norm
    assert enc.out_spec == FeatureSpec(channels={"C5": 16}, strides={"C5": 32})


def test_forward_shapes_and_extra_pass_through():
    enc = _encoder(in_features=("C4", "C5"))
    assert enc.out_spec == FeatureSpec(channels={"C4": 16, "C5": 16}, strides={"C4": 16, "C5": 32})
    out = enc(FeatureMaps(feature_maps=_maps(2, 4, 6, 2, 3), extra={"foo": 1}))
    assert set(out.feature_maps) == {"C4", "C5"}
    assert out.feature_maps["C4"].shape == (2, 16, 4, 6)
    assert out.feature_maps["C5"].shape == (2, 16, 2, 3)
    assert out.extra == {"foo": 1}


def test_level_embedding_only_with_several_levels_and_added_per_level():
    assert _encoder().level_embed is None
    enc = _encoder(in_features=("C4", "C5"), num_layers=0)    # no layers: output = projection + level embed
    assert enc.level_embed.shape == (2, 16)
    maps = _maps(1, 2, 3, 1, 2)
    out = enc(FeatureMaps(feature_maps=maps))
    for lvl, name in enumerate(["C4", "C5"]):
        expected = enc.input_proj[name](maps[name]) + enc.level_embed[lvl][None, :, None, None]
        torch.testing.assert_close(out.feature_maps[name], expected)


def test_pre_norm_has_final_layer_norm():
    enc = _encoder(normalize_before=True)
    assert isinstance(enc.norm, nn.LayerNorm)
    out = enc(FeatureMaps(feature_maps=_maps(1, 2, 2, 2, 2))).feature_maps["C5"]
    torch.testing.assert_close(out.mean(1), torch.zeros(1, 2, 2), atol=1e-5, rtol=0)


def test_construction_errors():
    with pytest.raises(ValueError, match="not in the in_spec"):
        _encoder(in_features=("P3",))
    with pytest.raises(ValueError, match="divisible by 4"):
        _encoder(d_model=18, num_heads=2)
    with pytest.raises(ValueError):
        Neck.create("detr_encoder", FeatureSpec(embed_dim=8))


@pytest.mark.parametrize("normalize_before", [False, True])
@pytest.mark.parametrize("in_features", [(), ("C3", "C4", "C5")])
def test_padding_invariance(normalize_before, in_features):
    """An image padded into a bigger batch gives the same values on its real cells."""
    enc = _encoder(in_features=in_features, normalize_before=normalize_before)
    torch.manual_seed(1)
    # image A: 40 x 72 px -> C3 5 x 9, C4 3 x 5, C5 2 x 3 cells; the batch is padded to 64 x 96
    alone = _maps(1, 3, 5, 2, 3, 5, 9)
    batch = _maps(2, 4, 6, 2, 3, 8, 12)                        # random everywhere, also on A's padding
    for name, m in alone.items():
        h, w = m.shape[-2:]
        batch[name][1, :, :h, :w] = m[0]
    sizes = [(64, 96), (40, 72)]

    out_alone = enc(FeatureMaps(feature_maps=alone, extra={"image_sizes": [(40, 72)]})).feature_maps
    out_batch = enc(FeatureMaps(feature_maps=batch, extra={"image_sizes": sizes})).feature_maps
    for name in enc.feature_names:
        h, w = out_alone[name].shape[-2:]
        torch.testing.assert_close(out_batch[name][1:, :, :h, :w], out_alone[name], rtol=1e-4, atol=1e-5)

    # sanity: without image_sizes the padding does leak into the real cells
    leaked = enc(FeatureMaps(feature_maps=batch)).feature_maps["C5"]
    assert not torch.allclose(leaked[1:, :, :2, :3], out_alone["C5"], atol=1e-3)
