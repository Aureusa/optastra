import pytest
import torch

import optastra
from optastra.core.factory import FACTORIES
from optastra.decoders import Decoder, QueryDecoder, QuerySet, inverse_sigmoid, keys
from optastra.nn.features import FeatureMaps, FeatureSpec


D = 16
QUERY_TYPES = ["vanilla", "conditional", "dab"]
PRESETS = {"detr_decoder": "vanilla", "conditional_detr_decoder": "conditional", "dab_detr_decoder": "dab"}


def _spec() -> FeatureSpec:
    # P4 already at d_model (identity projection), P5 is projected
    return FeatureSpec(channels={"P4": D, "P5": 12}, strides={"P4": 16, "P5": 32})


def _decoder(query_type="vanilla", **overrides) -> QueryDecoder:
    torch.manual_seed(0)
    kw = dict(d_model=D, num_queries=5, num_layers=3, num_heads=2, dim_feedforward=32, dropout=0.0,
              query_type=query_type)
    kw.update(overrides)
    return Decoder.create("detr_decoder", _spec(), **kw).eval()


def _features(B=2, sizes=None, gen_seed=1):
    g = torch.Generator().manual_seed(gen_seed)
    maps = {"P4": torch.randn(B, D, 4, 6, generator=g), "P5": torch.randn(B, 12, 2, 3, generator=g)}
    return FeatureMaps(feature_maps=maps, extra={keys.IMAGE_SIZES: sizes} if sizes is not None else {})


# --- registration / construction ------------------------------------------------------

def test_decoder_family_registered_and_exported():
    assert FACTORIES["decoder"] is Decoder
    assert optastra.Decoder is Decoder
    assert set(PRESETS) <= set(Decoder.list_all())


@pytest.mark.parametrize("name, query_type", PRESETS.items())
def test_presets(name, query_type):
    cfg = Decoder.get_default_config(name)
    assert cfg.query_type == query_type
    assert cfg.num_queries == (100 if query_type == "vanilla" else 300)
    assert cfg.normalize_before is False
    dec = Decoder.create(name, _spec(), d_model=D, num_heads=2, num_layers=1, dim_feedforward=16)
    assert isinstance(dec, QueryDecoder) and dec.cfg.query_type == query_type


@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_out_spec_and_memory_levels(query_type):
    dec = _decoder(query_type)
    assert dec.out_spec == FeatureSpec(embed_dim=D, num_tokens=5)
    assert dec.feature_names == ["P4", "P5"]                     # all maps, finest first
    assert isinstance(dec.input_proj["P4"], torch.nn.Identity)
    assert dec.level_embed.shape == (2, D)
    assert dec.uses_reference_points == (query_type != "vanilla")
    single = _decoder(query_type, in_features=("P5",))
    assert single.level_embed is None and single.feature_names == ["P5"]


def test_construction_errors():
    with pytest.raises(ValueError, match="query_type"):
        _decoder("deformable")
    with pytest.raises(ValueError, match="divisible by 4"):
        _decoder(d_model=18, num_heads=2)
    with pytest.raises(ValueError, match="not in the in_spec"):
        _decoder(in_features=("C5",))
    with pytest.raises(ValueError):
        Decoder.create("detr_decoder", FeatureSpec(embed_dim=D))   # needs feature maps


def test_inverse_sigmoid():
    x = torch.tensor([0.1, 0.5, 0.9])
    torch.testing.assert_close(inverse_sigmoid(x).sigmoid(), x)
    assert inverse_sigmoid(torch.tensor([0.0])).item() == pytest.approx(torch.logit(torch.tensor(1e-5)).item())


# --- forward contract --------------------------------------------------------------------

@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_forward_contract(query_type):
    dec = _decoder(query_type)
    sizes = [(64, 96), (50, 70)]
    out = dec(_features(sizes=sizes))
    assert out.patch_tokens.shape == (2, 5, D)
    layers = out.extra[keys.LAYER_TOKENS]
    assert len(layers) == 3
    assert all(t.shape == (2, 5, D) for t in layers)
    assert torch.equal(layers[-1], out.patch_tokens)
    torch.testing.assert_close(layers[0].mean(-1), torch.zeros(2, 5), atol=1e-5, rtol=0)   # final norm on each
    assert out.extra[keys.QUERY_MASK].dtype == torch.bool and out.extra[keys.QUERY_MASK].all()
    assert out.extra[keys.IMAGE_SIZES] == sizes
    if query_type == "vanilla":
        assert keys.REFERENCE_POINTS not in out.extra
    else:
        refs = out.extra[keys.REFERENCE_POINTS]
        assert refs.shape == (2, 5, 2)
        assert ((refs > 0) & (refs < 1)).all()
        torch.testing.assert_close(refs[0], refs[1])                # learned queries: same for every image


def test_dab_reference_points_are_the_anchors():
    dec = _decoder("dab")
    refs = dec(_features()).extra[keys.REFERENCE_POINTS]
    torch.testing.assert_close(refs[0], dec.anchors.sigmoid())


def test_conditional_reference_points_come_from_query_embedding():
    dec = _decoder("conditional")
    refs = dec(_features()).extra[keys.REFERENCE_POINTS]
    torch.testing.assert_close(refs[0], dec.ref_point_head(dec.query_embed.weight).sigmoid())


# --- external queries ----------------------------------------------------------------------

def _external(B=2, K=3, seed=2):
    g = torch.Generator().manual_seed(seed)
    return QuerySet(
        content=torch.randn(B, K, D, generator=g),
        reference_points=torch.rand(B, K, 2, generator=g) * 0.8 + 0.1,
        mask=torch.tensor([[True, True, False], [True, False, False]])[:B, :K],
    )


@pytest.mark.parametrize("query_type", ["conditional", "dab"])
def test_external_queries_are_placed_first(query_type):
    dec = _decoder(query_type, external_queries=(query_type == "conditional"))
    q = _external()
    out = dec(_features(), q)
    assert out.patch_tokens.shape == (2, 8, D)
    assert all(t.shape == (2, 8, D) for t in out.extra[keys.LAYER_TOKENS])
    refs = out.extra[keys.REFERENCE_POINTS]
    torch.testing.assert_close(refs[:, :3], q.reference_points)
    torch.testing.assert_close(refs[:, 3:], dec(_features()).extra[keys.REFERENCE_POINTS])
    qmask = out.extra[keys.QUERY_MASK]
    assert torch.equal(qmask[:, :3], q.mask) and qmask[:, 3:].all()


@pytest.mark.parametrize("query_type", ["conditional", "dab"])
def test_padded_external_queries_do_not_influence_real_queries(query_type):
    dec = _decoder(query_type, external_queries=(query_type == "conditional"))
    feats = _features(sizes=[(64, 96), (50, 70)])
    q = _external()
    out = dec(feats, q).patch_tokens

    pad = ~q.mask
    content, refs = q.content.clone(), q.reference_points.clone()
    content[pad] = content[pad] * 5 + torch.linspace(-3, 5, D)       # non-constant: LayerNorm can't hide it
    refs[pad] = torch.tensor([0.95, 0.03])
    out2 = dec(feats, QuerySet(content=content, reference_points=refs, mask=q.mask)).patch_tokens

    real = torch.cat([q.mask, torch.ones(2, 5, dtype=torch.bool)], 1)
    torch.testing.assert_close(out2[real], out[real], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(out2[~real], out[~real], atol=1e-3)     # the padded ones themselves did change


@pytest.mark.parametrize("query_type", ["conditional", "dab"])
def test_real_external_queries_do_influence_learned_queries(query_type):
    dec = _decoder(query_type, external_queries=(query_type == "conditional"))
    feats = _features()
    q = _external()
    content = q.content.clone()
    content[0, 0] += torch.linspace(-3, 5, D)
    a = dec(feats, q).patch_tokens[0, 3:]
    b = dec(feats, QuerySet(content=content, reference_points=q.reference_points, mask=q.mask)).patch_tokens[0, 3:]
    assert not torch.allclose(a, b, atol=1e-4)


def test_external_query_errors():
    feats, q = _features(), _external()
    with pytest.raises(ValueError, match="[Vv]anilla"):
        _decoder("vanilla")(feats, q)
    with pytest.raises(ValueError, match="external_queries=True"):
        _decoder("conditional")(feats, q)
    with pytest.raises(ValueError, match="reference_points"):
        _decoder("dab")(feats, QuerySet(content=q.content))
    # an empty QuerySet is the same as none
    dec = _decoder("vanilla")
    torch.testing.assert_close(dec(feats, QuerySet()).patch_tokens, dec(feats).patch_tokens)


def test_external_queries_without_content_or_mask_default_to_zeros_and_all_real():
    dec = _decoder("dab")
    q = _external()
    out = dec(_features(), QuerySet(reference_points=q.reference_points))
    assert out.extra[keys.QUERY_MASK].all()
    ref = dec(_features(), QuerySet(content=torch.zeros(2, 3, D), reference_points=q.reference_points))
    torch.testing.assert_close(out.patch_tokens, ref.patch_tokens)


# --- padding invariance of the memory ---------------------------------------------------------

@pytest.mark.parametrize("query_type", QUERY_TYPES)
@pytest.mark.parametrize("normalize_before", [False, True])
def test_memory_padding_does_not_influence_outputs(query_type, normalize_before):
    dec = _decoder(query_type, normalize_before=normalize_before)
    # image A: 40 x 72 px -> P4 3 x 5, P5 2 x 3 cells; alone, and padded into a 64 x 96 batch
    alone = _features(B=1, sizes=[(40, 72)], gen_seed=3)
    alone.feature_maps["P4"] = alone.feature_maps["P4"][..., :3, :5]
    batch = _features(B=2, sizes=[(64, 96), (40, 72)], gen_seed=4)
    batch.feature_maps["P4"][1, :, :3, :5] = alone.feature_maps["P4"][0]
    batch.feature_maps["P5"][1] = alone.feature_maps["P5"][0]

    out_alone = dec(alone)
    out_batch = dec(batch)
    for a, b in zip(out_alone.extra[keys.LAYER_TOKENS], out_batch.extra[keys.LAYER_TOKENS]):
        torch.testing.assert_close(b[1:], a, rtol=1e-4, atol=1e-5)

    # the padding cells really were different (else the test proves nothing)
    pad = batch.feature_maps["P4"].clone()
    pad[1, :, 3:, :] += torch.linspace(-3, 5, D)[:, None, None]
    pad[1, :, :, 5:] += torch.linspace(-3, 5, D)[:, None, None]
    batch.feature_maps["P4"] = pad
    torch.testing.assert_close(dec(batch).patch_tokens[1:], out_alone.patch_tokens, rtol=1e-4, atol=1e-5)


# --- gradients (DDP: every parameter must be used) ----------------------------------------------

@pytest.mark.parametrize("query_type", QUERY_TYPES)
def test_every_parameter_receives_a_gradient(query_type):
    dec = _decoder(query_type, external_queries=(query_type == "conditional")).train()
    queries = _external() if query_type != "vanilla" else None
    out = dec(_features(sizes=[(64, 96), (50, 70)]), queries)
    g = torch.Generator().manual_seed(0)
    # random weights: the plain sum of a LayerNorm output has zero gradient
    loss = sum((t * torch.randn(t.shape, generator=g)).sum() for t in out.extra[keys.LAYER_TOKENS])
    loss.backward()
    assert [n for n, p in dec.named_parameters() if p.grad is None] == []
    zero = [n for n, p in dec.named_parameters() if not torch.any(p.grad != 0)]
    # learned queries start with zero content (as in DETR), so the first self-attention's value
    # weight multiplies zeros: used (DDP is fine) but its gradient is exactly zero. External
    # queries carry content, so then every gradient is non-zero.
    assert zero == ([] if queries is not None else ["layers.0.self_attn.v.weight"])
