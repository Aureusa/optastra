import pytest
import torch

from optastra import Backbone, Head, build_sequential_model
from optastra.heads import RegressionHead, RegressionHeadConfig
from optastra.heads.regression import (
    BBoxRegressionHead,
    BBoxRegressionHeadConfig,
    vanilla_box_regression_head,
)
from optastra.nn.features import FeatureMaps, FeatureSpec


def test_bbox_regression_head_forward_returns_values_with_expected_shape():
    in_spec = FeatureSpec(embed_dim=256)
    cfg = BBoxRegressionHeadConfig(hidden_features=128, num_layers=2, box_dim=4, class_agnostic=True)
    head = BBoxRegressionHead(in_spec=in_spec, cfg=cfg)

    features = FeatureMaps(pooled=torch.randn(5, 256))
    out = head(features)

    assert out.values.shape == (5, 4)
    assert out.logits is None


def test_bbox_regression_head_can_be_class_specific():
    in_spec = FeatureSpec(embed_dim=128)
    cfg = BBoxRegressionHeadConfig(
        hidden_features=64,
        num_layers=2,
        box_dim=4,
        class_agnostic=False,
        num_classes=6,
    )
    head = BBoxRegressionHead(in_spec=in_spec, cfg=cfg)

    features = FeatureMaps(pooled=torch.randn(3, 128))
    out = head(features)

    assert out.values.shape == (3, 24)


def test_vanilla_box_regression_head_factory_returns_head_instance():
    in_spec = FeatureSpec(embed_dim=64)
    cfg = BBoxRegressionHeadConfig(hidden_features=32, num_layers=2)

    head = vanilla_box_regression_head(in_spec=in_spec, cfg=cfg)

    assert isinstance(head, BBoxRegressionHead)


def test_bbox_regression_head_requires_embed_dim_and_pooled():
    import pytest

    with pytest.raises(ValueError, match="embed_dim"):
        BBoxRegressionHead(in_spec=FeatureSpec(channels={"C5": 8}, strides={"C5": 32}), cfg=BBoxRegressionHeadConfig())

    head = BBoxRegressionHead(in_spec=FeatureSpec(embed_dim=8), cfg=BBoxRegressionHeadConfig(hidden_features=4))
    with pytest.raises(ValueError, match="pooled"):
        head(FeatureMaps(feature_maps={"C5": torch.randn(1, 8, 2, 2)}))


# --- RegressionHead (continuous per-image values) ---------------------------

def test_regression_head_is_registered_and_outputs_values_per_output():
    assert "vanilla_regression_head" in Head.list_all()
    head = Head.create("vanilla_regression_head", in_spec=FeatureSpec(embed_dim=32), num_outputs=3, hidden_features=16)
    assert isinstance(head, RegressionHead)
    out = head(FeatureMaps(pooled=torch.randn(5, 32)))
    assert out.values.shape == (5, 3)
    assert out.logits is None


def test_regression_head_single_layer_is_an_affine_map_of_pooled_features():
    head = RegressionHead(FeatureSpec(embed_dim=4), RegressionHeadConfig(num_layers=1, num_outputs=2))
    linear = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear) == 1
    x = torch.randn(3, 4)
    torch.testing.assert_close(head(FeatureMaps(pooled=x)).values, linear[0](x))


def test_regression_head_needs_pooled_features():
    spatial = Backbone.create("resnet18").out_spec
    with pytest.raises(ValueError, match="embed_dim"):
        Head.create("vanilla_regression_head", in_spec=spatial)
    head = Head.create("vanilla_regression_head", in_spec=FeatureSpec(embed_dim=8))
    with pytest.raises(ValueError, match="pooling neck"):
        head(FeatureMaps(pooled=None))


def test_regression_head_composes_with_backbone_and_neck():
    model = build_sequential_model("resnet18", ["global_avg_pool"], ("vanilla_regression_head", {"num_outputs": 2}))
    assert model(torch.randn(2, 3, 32, 32)).values.shape == (2, 2)
