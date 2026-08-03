import torch
import pytest

from optastra.heads.classification import (
    ClassificationHead,
    ClassificationHeadConfig,
    vanilla_classification_head,
)
from optastra.nn.features import FeatureMaps, FeatureSpec


def test_classification_head_forward_returns_logits_with_expected_shape():
    in_spec = FeatureSpec(embed_dim=512)
    cfg = ClassificationHeadConfig(hidden_features=256, num_layers=2, num_classes=10)
    head = ClassificationHead(in_spec=in_spec, cfg=cfg)

    features = FeatureMaps(pooled=torch.randn(4, 512))

    out = head(features)

    assert out.logits.shape == (4, 10)
    assert out.values is None
    assert out.boxes is None
    assert out.masks is None


def test_classification_head_uses_num_classes_from_config():
    in_spec = FeatureSpec(embed_dim=256)
    cfg = ClassificationHeadConfig(hidden_features=128, num_layers=3, num_classes=7)
    head = ClassificationHead(in_spec=in_spec, cfg=cfg)

    features = FeatureMaps(pooled=torch.randn(2, 256))

    out = head(features)

    assert out.logits.shape == (2, 7)
    assert head.num_classes == 7


def test_classification_head_raises_when_pooled_features_are_missing():
    in_spec = FeatureSpec(embed_dim=64)
    cfg = ClassificationHeadConfig(hidden_features=32, num_layers=2, num_classes=3)
    head = ClassificationHead(in_spec=in_spec, cfg=cfg)

    features = FeatureMaps(feature_maps={"P5": torch.randn(1, 64, 7, 7)})

    with pytest.raises(TypeError):
        head(features)


def test_vanilla_classification_head_factory_returns_classification_head():
    in_spec = FeatureSpec(embed_dim=64)
    cfg = ClassificationHeadConfig(hidden_features=32, num_layers=2, num_classes=3)
    head = vanilla_classification_head(in_spec=in_spec, cfg=cfg)

    assert isinstance(head, ClassificationHead)
