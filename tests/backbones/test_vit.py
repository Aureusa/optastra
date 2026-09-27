import pytest
import torch

from optastra import build_sequential_model
from optastra.backbones import Backbone
from optastra.necks import Neck
from optastra.nn.blocks.transformer.pos_embed import get_2d_sincos_pos_embed, interpolate_pos_embed

# Small ViT so the tests run fast on CPU: 32x32 images, 8x8 patches -> 4x4 grid.
SMALL = dict(img_size=32, patch_size=8, depth=2)


def _vit(**overrides):
    return Backbone.create("vit_tiny", **{**SMALL, **overrides}).eval()


def test_sincos_pos_embed_is_not_overwritten_by_random_init():
    torch.manual_seed(0)
    model = _vit(pos_embed_type="sinusoidal")
    expected = get_2d_sincos_pos_embed(192, 4, cls_token=True)
    torch.testing.assert_close(model.pos_embed.pos_embed, expected)


def test_sincos_vit_reloads_to_identical_outputs():
    # The sin-cos table is a non-persistent buffer: it is not in the checkpoint,
    # every instance rebuilds the same table, so a reloaded model is identical.
    torch.manual_seed(0)
    model = _vit(pos_embed_type="sinusoidal")
    assert not any("pos_embed" in key for key in model.state_dict())

    torch.manual_seed(1)
    reloaded = _vit(pos_embed_type="sinusoidal")
    reloaded.load_state_dict(model.state_dict())

    x = torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(reloaded(x).pooled, model(x).pooled)


def test_learned_pos_embed_is_persisted_and_reloaded():
    torch.manual_seed(0)
    model = _vit()
    torch.manual_seed(1)
    reloaded = _vit()
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded.pos_embed.pos_embed, model.pos_embed.pos_embed)


def test_load_state_dict_interpolates_learned_pos_embed_across_img_size():
    small = _vit(img_size=32)    # 4x4 grid
    large = _vit(img_size=64)    # 8x8 grid

    state = small.state_dict()
    large.load_state_dict(state)

    expected = interpolate_pos_embed(state["pos_embed.pos_embed"], 4, 8, num_prefix_tokens=1)
    assert large.pos_embed.pos_embed.shape == (1, 1 + 64, 192)
    torch.testing.assert_close(large.pos_embed.pos_embed, expected)
    # the cls position is copied, not interpolated
    torch.testing.assert_close(large.pos_embed.pos_embed[:, 0], state["pos_embed.pos_embed"][:, 0])


@pytest.mark.parametrize("pos_embed_type", ["learned", "sinusoidal"])
def test_vit_runs_at_other_input_sizes(pos_embed_type):
    model = _vit(pos_embed_type=pos_embed_type)
    out = model(torch.randn(2, 3, 64, 48))

    assert out.patch_tokens.shape == (2, 8 * 6, 192)
    assert out.feature_maps["C3"].shape == (2, 192, 8, 6)


def test_vit_dynamic_size_matches_sincos_table_for_that_size():
    model = _vit(pos_embed_type="sinusoidal", depth=0)   # no blocks: output = norm(tokens + pos)
    model.norm = torch.nn.Identity()
    x = torch.randn(1, 3, 48, 64)
    tokens = model.patch_embed(x)
    cls = model.cls_token.expand(1, -1, -1)
    expected = torch.cat([cls, tokens], dim=1) + get_2d_sincos_pos_embed(192, (6, 8), cls_token=True)

    out = model(x)
    torch.testing.assert_close(out.cls_token, expected[:, 0])
    torch.testing.assert_close(out.patch_tokens, expected[:, 1:])


def test_vit_patch_tokens_reshape_to_row_major_spatial_map():
    model = _vit()
    out = model(torch.randn(1, 3, 32, 64))   # 4 x 8 grid
    spatial = out.feature_maps["C3"]
    # token t sits at row t // 8, column t % 8
    torch.testing.assert_close(spatial[0, :, 1, 3], out.patch_tokens[0, 1 * 8 + 3])
    assert model.out_spec.channels == {"C3": 192}
    assert model.out_spec.strides == {"C3": 8}


def test_vit_default_patch16_exposes_stride16_c4_map():
    model = Backbone.create("vit_tiny", depth=1).eval()
    assert model.out_spec.strides == {"C4": 16}
    out = model(torch.randn(1, 3, 224, 224))
    assert out.feature_maps["C4"].shape == (1, 192, 14, 14)


def test_vit_without_cls_token_pools_patch_tokens():
    model = _vit(cls_token=False)
    out = model(torch.randn(2, 3, 32, 32))

    assert out.cls_token is None
    assert out.patch_tokens.shape == (2, 16, 192)
    torch.testing.assert_close(out.pooled, out.patch_tokens.mean(dim=1))


def test_vit_with_cls_token_pools_cls_token():
    model = _vit()
    out = model(torch.randn(2, 3, 32, 32))
    torch.testing.assert_close(out.pooled, out.cls_token)


def test_vit_global_avg_pool_equals_mean_patch_token():
    model = build_sequential_model(
        backbone=("vit_tiny", SMALL),
        necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": 5}),
    ).eval()
    x = torch.randn(2, 3, 32, 32)
    features = model[0](x)
    pooled = model[1](features).pooled
    torch.testing.assert_close(pooled, features.patch_tokens.mean(dim=1))
    assert model(x).logits.shape == (2, 5)


def test_vit_token_pool_classification():
    model = build_sequential_model(
        backbone=("vit_tiny", SMALL),
        necks=[("token_pool", {"method": "max"})],
        head=("vanilla_classification_head", {"num_classes": 5}),
    ).eval()
    x = torch.randn(2, 3, 32, 32)
    features = model[0](x)
    torch.testing.assert_close(model[1](features).pooled, features.patch_tokens.amax(dim=1))
    assert model(x).logits.shape == (2, 5)


def test_vit_cls_head_without_neck():
    model = build_sequential_model(
        backbone=("vit_tiny", SMALL),
        necks=[],
        head=("vanilla_classification_head", {"num_classes": 3}),
    )
    assert model(torch.randn(2, 3, 32, 32)).logits.shape == (2, 3)


def test_vit_fpn_builds_and_forwards():
    backbone = _vit()
    fpn = Neck.create("fpn", backbone.out_spec, out_channels=64)
    assert fpn.out_spec.strides == {"P3": 8}

    out = fpn(backbone(torch.randn(2, 3, 32, 32)))
    assert out.feature_maps["P3"].shape == (2, 64, 4, 4)
