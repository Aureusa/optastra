import pytest
import torch
import torch.nn as nn

from optastra.nn.blocks.convolution import ConvNormAct, LocalResponseNorm
from optastra.nn.blocks.transformer import MultiHeadSelfAttention, RotaryPosEmbed2D
from optastra.nn.layers import LayerNorm2d, StochasticDepth, drop_path_rates


def test_attention_matches_explicit_softmax_formula():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(dim=32, num_heads=4).eval()
    x = torch.randn(2, 7, 32)

    # the pre-SDPA implementation, written out by hand
    B, N, C = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, 4, 8).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    weights = ((q @ k.transpose(-2, -1)) * 8 ** -0.5).softmax(dim=-1)
    expected = attn.proj((weights @ v).transpose(1, 2).reshape(B, N, C))

    torch.testing.assert_close(attn(x), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("size", [5, 4, 3])
def test_local_response_norm_matches_torch(size):
    torch.manual_seed(0)
    x = torch.randn(2, 11, 5, 6) * 10
    ours = LocalResponseNorm(size=size, alpha=1e-2, beta=0.75, k=2.0)(x)
    reference = nn.LocalResponseNorm(size=size, alpha=1e-2, beta=0.75, k=2.0)(x)
    torch.testing.assert_close(ours, reference)


def test_conv_norm_act_layernorm_normalizes_channels():
    torch.manual_seed(0)
    block = ConvNormAct(in_channels=3, out_channels=8, kernel_size=3, norm="layernorm", activation=None)
    assert isinstance(block.norm, LayerNorm2d)

    out = block(torch.randn(2, 3, 5, 7))      # works for any H, W (nn.LayerNorm(8) on NCHW would not)
    torch.testing.assert_close(out.mean(dim=1), torch.zeros(2, 5, 7), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.var(dim=1, unbiased=False), torch.ones(2, 5, 7), atol=1e-3, rtol=0)


def test_conv_norm_act_groupnorm_groups():
    assert ConvNormAct(in_channels=3, out_channels=64, kernel_size=1, norm="groupnorm").norm.num_groups == 32
    assert ConvNormAct(in_channels=3, out_channels=24, kernel_size=1, norm="groupnorm").norm.num_groups == 8
    assert ConvNormAct(in_channels=3, out_channels=24, kernel_size=1, norm="groupnorm", num_groups=6).norm.num_groups == 6
    with pytest.raises(ValueError, match="must divide"):
        ConvNormAct(in_channels=3, out_channels=24, kernel_size=1, norm="groupnorm", num_groups=5)


def test_rotary_pos_embed_keeps_prefix_and_is_relative():
    torch.manual_seed(0)
    grid = 4
    rope = RotaryPosEmbed2D(head_dim=16, grid_size=grid)

    # same q (and same k) at every position -> q_i . k_j must depend only on the offset i - j
    q = torch.randn(16).expand(1, 1, 1 + grid * grid, 16).clone()
    k = torch.randn(16).expand(1, 1, 1 + grid * grid, 16).clone()
    q_rot, k_rot = rope.rotate(q), rope.rotate(k)

    assert q_rot.shape == q.shape
    torch.testing.assert_close(q_rot[:, :, 0], q[:, :, 0])                       # cls token untouched
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1))              # rotation keeps norms

    def score(pos_q, pos_k):  # (row, col) positions, +1 for the cls prefix
        i, j = 1 + pos_q[0] * grid + pos_q[1], 1 + pos_k[0] * grid + pos_k[1]
        return (q_rot[0, 0, i] * k_rot[0, 0, j]).sum()

    torch.testing.assert_close(score((0, 0), (1, 2)), score((2, 1), (3, 3)))
    torch.testing.assert_close(score((1, 0), (0, 0)), score((3, 2), (2, 2)))


def test_stochastic_depth_old_import_path_and_rates():
    from optastra.nn.blocks.transformer.stochastic_depth import StochasticDepth as OldPath
    assert OldPath is StochasticDepth
    assert drop_path_rates(0.3, 4) == pytest.approx([0.0, 0.1, 0.2, 0.3])

    layer = StochasticDepth(0.5).eval()
    x = torch.randn(4, 3)
    assert torch.equal(layer(x), x)
