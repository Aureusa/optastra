import torch

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
