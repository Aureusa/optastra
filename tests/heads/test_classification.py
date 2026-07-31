import torch
import pytest

from optastra.backbones.base import BackboneFeatures
from optastra.necks.base import NeckFeatures
from optastra.heads.classification import ClassificationHead, vanilla_classification_head


def test_classification_head_forward_with_backbone_features_avg_pooling():
    head = ClassificationHead(
        in_features=512,
        hidden_features=256,
        stage="C5",
        pooling="avg",
        num_layers=2,
        num_classes=10,
    )

    features = BackboneFeatures(
        feature_maps={"C5": torch.randn(4, 512, 7, 7)}
    )

    out = head(features)

    assert out.logits.shape == (4, 10)
    assert out.predictions.shape == (4, 10)
    assert torch.allclose(
        out.predictions.sum(dim=-1),
        torch.ones(4),
        atol=1e-5,
        rtol=1e-5,
    )


def test_classification_head_forward_with_neck_features_max_pooling():
    head = ClassificationHead(
        in_features=256,
        hidden_features=128,
        stage="P5",
        pooling="max",
        num_layers=3,
        num_classes=7,
    )

    features = NeckFeatures(
        feature_maps={"P5": torch.randn(2, 256, 14, 14)}
    )

    out = head(features)

    assert out.logits.shape == (2, 7)
    assert out.predictions.shape == (2, 7)


def test_classification_head_raises_on_missing_stage():
    head = ClassificationHead(
        in_features=512,
        hidden_features=256,
        stage="C5",
        pooling="avg",
        num_layers=2,
        num_classes=10,
    )

    features = BackboneFeatures(
        feature_maps={"C4": torch.randn(1, 512, 14, 14)}
    )

    with pytest.raises(AssertionError, match="Stage 'C5' not found"):
        head(features)


def test_vanilla_classification_head_factory_returns_classification_head():
    head = vanilla_classification_head(
        in_features=64,
        hidden_features=32,
        stage="C5",
        pooling="avg",
        num_layers=2,
        num_classes=3,
    )

    assert isinstance(head, ClassificationHead)
