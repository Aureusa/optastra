"""Masked / cross attention, DETR encoder and decoder layers, sine_embed_2d and
the feature-level flattening shared by the DETR encoder and query decoders."""
import math

import pytest
import torch
import torch.nn.functional as F

from optastra.nn.blocks.transformer import (
    ConditionalCrossAttention,
    CrossAttentionBlock,
    DETRDecoderLayer,
    DETREncoderLayer,
    MaskedTransformerBlock,
    MultiHeadAttention,
    flatten_feature_levels,
    padding_to_attn_mask,
    sine_embed_2d,
    unflatten_feature_levels,
)


def _split(x, H):
    B, L, D = x.shape
    return x.reshape(B, L, H, D // H).transpose(1, 2)


def _merge(x):
    B, H, L, d = x.shape
    return x.transpose(1, 2).reshape(B, L, H * d)


def _manual_mha(attn: MultiHeadAttention, query, key, value, mask=None):
    """softmax(q k^T / sqrt(d) + mask) v, written out by hand. Returns (output, weights)."""
    H, d = attn.num_heads, attn.head_dim
    q, k, v = _split(attn.q(query), H), _split(attn.k(key), H), _split(attn.v(value), H)
    logits = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else logits + mask
    weights = logits.softmax(-1)
    return attn.proj(_merge(weights @ v)), weights


# --- MultiHeadAttention -------------------------------------------------------------

def test_mha_matches_formula_with_separate_key_width():
    torch.manual_seed(0)
    attn = MultiHeadAttention(16, num_heads=4, kdim=6).eval()
    q, kv = torch.randn(2, 5, 16), torch.randn(2, 7, 6)
    expected, _ = _manual_mha(attn, q, kv, kv)
    torch.testing.assert_close(attn(q, kv, kv), expected, rtol=1e-5, atol=1e-5)


def test_mha_bool_mask_matches_formula_and_masked_keys_get_zero_weight():
    torch.manual_seed(0)
    attn = MultiHeadAttention(16, num_heads=2).eval()
    q, k, v = torch.randn(2, 4, 16), torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    key_mask = torch.tensor([[True, True, True, False, False, False],
                             [True, False, True, True, True, False]])
    mask = padding_to_attn_mask(key_mask)
    assert mask.shape == (2, 1, 1, 6)

    out = attn(q, k, v, mask)
    expected, weights = _manual_mha(attn, q, k, v, mask)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    assert (weights.masked_select(~mask.expand_as(weights)) == 0).all()

    # masked keys / values can be anything: the output does not change
    k2, v2 = k.clone(), v.clone()
    k2[~key_mask] = torch.randn(int((~key_mask).sum()), 16) * 100
    v2[~key_mask] = torch.randn(int((~key_mask).sum()), 16) * 100
    torch.testing.assert_close(attn(q, k2, v2, mask), out, rtol=1e-5, atol=1e-5)


def test_mha_full_per_head_bool_mask():
    torch.manual_seed(0)
    attn = MultiHeadAttention(8, num_heads=2).eval()
    q, k = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    mask = torch.rand(1, 2, 3, 5) > 0.5
    mask[..., 0] = True                                     # every row allows at least one key
    expected, _ = _manual_mha(attn, q, k, k, mask)
    torch.testing.assert_close(attn(q, k, k, mask), expected, rtol=1e-5, atol=1e-5)


def test_mha_additive_float_mask_matches_formula():
    torch.manual_seed(0)
    attn = MultiHeadAttention(8, num_heads=2).eval()
    q, k = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    bias = torch.randn(2, 2, 3, 5) * 3
    expected, _ = _manual_mha(attn, q, k, k, bias)
    torch.testing.assert_close(attn(q, k, k, bias), expected, rtol=1e-5, atol=1e-5)

    # -inf in a float mask works like False in a bool mask
    bias_inf = torch.zeros(2, 1, 1, 5)
    bias_inf[..., 3:] = float("-inf")
    bool_mask = torch.ones(2, 1, 1, 5, dtype=torch.bool)
    bool_mask[..., 3:] = False
    torch.testing.assert_close(attn(q, k, k, bias_inf), attn(q, k, k, bool_mask), rtol=1e-5, atol=1e-5)


def test_mha_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(10, num_heads=3)


def test_masked_transformer_block_adds_pos_only_to_queries_and_keys():
    torch.manual_seed(0)
    block = MaskedTransformerBlock(8, num_heads=2, mlp_ratio=2.0).eval()
    x, pos = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    mask = padding_to_attn_mask(torch.tensor([[True] * 5, [True, True, True, False, False]]))
    h = block.ln1(x)
    y = x + _manual_mha(block.attn, h + pos, h + pos, h, mask)[0]
    expected = y + block.mlp(block.ln2(y))
    torch.testing.assert_close(block(x, mask, pos), expected, rtol=1e-5, atol=1e-5)


def test_cross_attention_block_matches_formula():
    torch.manual_seed(0)
    block = CrossAttentionBlock(8, num_heads=2, mlp_ratio=2.0, kdim=6).eval()
    x, mem = torch.randn(2, 3, 8), torch.randn(2, 5, 6)
    qpos, mpos = torch.randn(2, 3, 8), torch.randn(2, 5, 6)
    q, kv = block.ln_q(x), block.ln_kv(mem)
    y = x + _manual_mha(block.attn, q + qpos, kv + mpos, kv)[0]
    expected = y + block.mlp(block.ln2(y))
    torch.testing.assert_close(block(x, mem, None, qpos, mpos), expected, rtol=1e-5, atol=1e-5)


# --- DETR encoder layer -----------------------------------------------------------------

def _zero_qk(attn):
    """q = k = 0 -> uniform attention whatever the positions."""
    with torch.no_grad():
        for lin in (attn.q, attn.k):
            lin.weight.zero_()
            lin.bias.zero_()


@pytest.mark.parametrize("normalize_before", [False, True])
def test_encoder_layer_matches_formula(normalize_before):
    torch.manual_seed(0)
    layer = DETREncoderLayer(16, 4, dim_feedforward=32, dropout=0.0, normalize_before=normalize_before).eval()
    src, pos = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    mask = padding_to_attn_mask(torch.tensor([[True] * 6, [True] * 4 + [False] * 2]))
    attn = lambda q, v: _manual_mha(layer.self_attn, q, q, v, mask)[0]
    ffn = lambda x: layer.ffn.linear2(F.relu(layer.ffn.linear1(x)))
    if normalize_before:
        h = layer.norm1(src)
        s = src + attn(h + pos, h)
        expected = s + ffn(layer.norm2(s))
    else:
        s = layer.norm1(src + attn(src + pos, src))
        expected = layer.norm2(s + ffn(s))
    torch.testing.assert_close(layer(src, pos, mask), expected, rtol=1e-5, atol=1e-5)


def test_encoder_layer_default_is_post_norm_and_output_is_normalised():
    torch.manual_seed(0)
    layer = DETREncoderLayer(16, 4, dim_feedforward=32, dropout=0.0).eval()
    assert layer.normalize_before is False
    out = layer(torch.randn(2, 5, 16) * 10 + 3)
    torch.testing.assert_close(out.mean(-1), torch.zeros(2, 5), atol=1e-5, rtol=0)


@pytest.mark.parametrize("normalize_before", [False, True])
def test_encoder_positions_enter_only_queries_and_keys(normalize_before):
    torch.manual_seed(0)
    layer = DETREncoderLayer(16, 4, dim_feedforward=32, dropout=0.0, normalize_before=normalize_before).eval()
    src, pos = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    assert not torch.allclose(layer(src, pos), layer(src, None), atol=1e-4)   # pos matters through q / k...
    _zero_qk(layer.self_attn)
    # ...and only there: with constant attention the value path must not see it
    torch.testing.assert_close(layer(src, pos), layer(src, None), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(layer(src, pos * 7 + 1), layer(src, None), rtol=1e-5, atol=1e-5)


# --- DETR decoder layer ------------------------------------------------------------------

def _ffn(layer, x):
    return layer.ffn.linear2(F.relu(layer.ffn.linear1(x)))


@pytest.mark.parametrize("normalize_before", [False, True])
def test_standard_decoder_layer_matches_formula(normalize_before):
    torch.manual_seed(0)
    layer = DETRDecoderLayer(16, 4, dim_feedforward=32, dropout=0.0, normalize_before=normalize_before).eval()
    tgt, qpos = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    mem, pos = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    self_mask = torch.tensor([True, True, False])[None, None, None, :] | torch.eye(3, dtype=torch.bool)
    cross_mask = padding_to_attn_mask(torch.tensor([[True] * 7, [True] * 5 + [False] * 2]))
    sa = lambda q, v: _manual_mha(layer.self_attn, q, q, v, self_mask)[0]
    ca = lambda q: _manual_mha(layer.cross_attn, q + qpos, mem + pos, mem, cross_mask)[0]
    if normalize_before:
        h = layer.norm1(tgt)
        t = tgt + sa(h + qpos, h)
        t = t + ca(layer.norm2(t))
        expected = t + _ffn(layer, layer.norm3(t))
    else:
        t = layer.norm1(tgt + sa(tgt + qpos, tgt))
        t = layer.norm2(t + ca(t))
        expected = layer.norm3(t + _ffn(layer, t))
    out = layer(tgt, mem, qpos, pos, self_mask, cross_mask)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("normalize_before", [False, True])
def test_standard_decoder_positions_enter_only_queries_and_keys(normalize_before):
    torch.manual_seed(0)
    layer = DETRDecoderLayer(16, 4, dim_feedforward=32, dropout=0.0, normalize_before=normalize_before).eval()
    tgt, qpos = torch.randn(2, 3, 16), torch.randn(2, 3, 16)
    mem, pos = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    assert not torch.allclose(layer(tgt, mem, qpos, pos), layer(tgt, mem), atol=1e-4)
    _zero_qk(layer.self_attn)
    _zero_qk(layer.cross_attn)
    torch.testing.assert_close(layer(tgt, mem, qpos, pos), layer(tgt, mem), rtol=1e-5, atol=1e-5)


def test_decoder_layer_argument_errors():
    with pytest.raises(ValueError, match="cross_attention"):
        DETRDecoderLayer(16, 4, cross_attention="deformable")
    layer = DETRDecoderLayer(16, 4, cross_attention="conditional").eval()
    with pytest.raises(ValueError, match="query_sine_embed"):
        layer(torch.randn(1, 2, 16), torch.randn(1, 3, 16), torch.randn(1, 2, 16), torch.randn(1, 3, 16))
    with pytest.raises(ValueError, match="divisible"):
        ConditionalCrossAttention(10, 3)


# --- Conditional cross-attention -----------------------------------------------------------

def _manual_conditional(m: ConditionalCrossAttention, tgt, memory, query_pos, pos, sine, mask=None):
    H, d = m.num_heads, m.head_dim
    q, k, k_pos = m.q_content(tgt), m.k_content(memory), m.k_pos(pos)
    if m.first_layer:
        q = q + m.q_pos(query_pos)
        k = k + k_pos
    qs = m.q_sine(sine)
    outs = []
    for h in range(H):                                      # one head at a time, explicit concatenation
        sl = slice(h * d, (h + 1) * d)
        qh = torch.cat([q[..., sl], qs[..., sl]], -1)       # (B, M, 2d)
        kh = torch.cat([k[..., sl], k_pos[..., sl]], -1)    # (B, L, 2d)
        logits = qh @ kh.transpose(-2, -1) / math.sqrt(2 * d)
        if mask is not None:
            logits = logits.masked_fill(~mask[:, 0], float("-inf"))
        outs.append(logits.softmax(-1) @ m.v(memory)[..., sl])
    return m.proj(torch.cat(outs, -1))


@pytest.mark.parametrize("num_heads", [1, 2])
@pytest.mark.parametrize("first_layer", [True, False])
def test_conditional_cross_attention_matches_manual_concatenation(num_heads, first_layer):
    torch.manual_seed(0)
    m = ConditionalCrossAttention(8, num_heads, first_layer=first_layer).eval()
    assert (m.q_pos is not None) == first_layer
    tgt, qpos, sine = torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    mem, pos = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    mask = padding_to_attn_mask(torch.tensor([[True] * 5, [True, True, False, False, False]]))
    out = m(tgt, mem, qpos, pos, sine, mask)
    torch.testing.assert_close(out, _manual_conditional(m, tgt, mem, qpos, pos, sine, mask), rtol=1e-5, atol=1e-5)


def test_conditional_cross_attention_later_layers_ignore_query_pos():
    torch.manual_seed(0)
    m = ConditionalCrossAttention(8, 2, first_layer=False).eval()
    tgt, sine, mem, pos = torch.randn(1, 3, 8), torch.randn(1, 3, 8), torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    torch.testing.assert_close(m(tgt, mem, torch.randn(1, 3, 8), pos, sine), m(tgt, mem, None, pos, sine))


def test_conditional_decoder_layer_matches_formula_post_norm():
    torch.manual_seed(0)
    layer = DETRDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, cross_attention="conditional",
                             first_layer=True).eval()
    tgt, qpos, sine = torch.randn(2, 3, 8), torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    mem, pos = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    t = layer.norm1(tgt + _manual_mha(layer.self_attn, tgt + qpos, tgt + qpos, tgt)[0])
    t = layer.norm2(t + _manual_conditional(layer.cross_attn, t, mem, qpos, pos, sine))
    expected = layer.norm3(t + _ffn(layer, t))
    out = layer(tgt, mem, qpos, pos, query_sine_embed=sine)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


# --- sine_embed_2d ------------------------------------------------------------------------

def test_sine_embed_2d_hand_values():
    # dim 8 -> 2 frequencies per axis, t_k = 10000 ** (k / 2) = [1, 100]; c = coord * 2 pi
    out = sine_embed_2d(torch.tensor([0.25, 0.5]), 8)
    cx, cy = 0.25 * 2 * math.pi, 0.5 * 2 * math.pi
    expected = torch.tensor([
        math.sin(cy), math.sin(cy / 100), math.cos(cy), math.cos(cy / 100),   # y first
        math.sin(cx), math.sin(cx / 100), math.cos(cx), math.cos(cx / 100),
    ])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    assert out[4].item() == pytest.approx(1.0) and out[0].item() == pytest.approx(0.0, abs=1e-6)


def test_sine_embed_2d_batches_and_custom_scale_and_rejects_bad_dim():
    pts = torch.rand(3, 4, 2)
    out = sine_embed_2d(pts, 16)
    assert out.shape == (3, 4, 16)
    torch.testing.assert_close(out[1, 2], sine_embed_2d(pts[1, 2], 16))
    torch.testing.assert_close(sine_embed_2d(pts, 16, scale=1.0), sine_embed_2d(pts / (2 * math.pi), 16))
    with pytest.raises(ValueError, match="divisible by 4"):
        sine_embed_2d(pts, 6)


# --- flatten_feature_levels -----------------------------------------------------------------

def test_flatten_feature_levels_padding_mask_and_centres_two_levels():
    torch.manual_seed(0)
    D = 8
    # padded batch of 32 x 40 pixels; image 1 is only 21 x 13
    maps = [torch.randn(2, D, 8, 10), torch.randn(2, D, 4, 5)]
    strides = [4, 8]
    sizes = [(32, 40), (21, 13)]
    flat = flatten_feature_levels(maps, strides, sizes)

    assert flat.shapes == [(8, 10), (4, 5)]
    assert flat.memory.shape == (2, 100, D) and flat.pos.shape == (2, 100, D)
    assert flat.mask.shape == (2, 100) and flat.centres.shape == (2, 100, 2)

    exp_mask = torch.zeros(2, 100, dtype=torch.bool)
    exp_centres = torch.zeros(2, 100, 2)
    for b, (ih, iw) in enumerate(sizes):
        idx = 0
        for (h, w), s in zip(flat.shapes, strides):
            for i in range(h):
                for j in range(w):
                    exp_mask[b, idx] = i < math.ceil(ih / s) and j < math.ceil(iw / s)
                    exp_centres[b, idx] = torch.tensor([(j + 0.5) * s / iw, (i + 0.5) * s / ih])
                    idx += 1
    assert torch.equal(flat.mask, exp_mask)
    assert flat.mask[0].all()
    # image 1: level 0 has 6 x 4 real cells (ceil(21/4), ceil(13/4)), level 1 has 3 x 2
    assert int(flat.mask[1, :80].sum()) == 24 and int(flat.mask[1, 80:].sum()) == 6
    torch.testing.assert_close(flat.centres, exp_centres)
    torch.testing.assert_close(flat.pos, sine_embed_2d(exp_centres, D))
    torch.testing.assert_close(flat.memory, torch.cat([m.flatten(2).transpose(1, 2) for m in maps], 1))

    for got, m in zip(unflatten_feature_levels(flat.memory, flat.shapes), maps):
        assert torch.equal(got, m)


def test_flatten_feature_levels_default_sizes_and_level_embed():
    torch.manual_seed(0)
    maps = [torch.randn(1, 4, 2, 3), torch.randn(1, 4, 1, 2)]
    embed = torch.randn(2, 4)
    flat = flatten_feature_levels(maps, [8, 16], level_embed=embed)
    assert flat.mask.all()
    # no sizes: the image is the level-0 map times its stride (16 x 24)
    torch.testing.assert_close(flat.centres[0, :6], torch.tensor(
        [[(j + 0.5) / 3, (i + 0.5) / 2] for i in range(2) for j in range(3)]))
    torch.testing.assert_close(flat.memory[0, :6], maps[0][0].flatten(1).T + embed[0])
    torch.testing.assert_close(flat.memory[0, 6:], maps[1][0].flatten(1).T + embed[1])
