import torch

from optastra.heads.mask import MaskRCNNHead, MaskRCNNHeadConfig, mask_rcnn_head
from optastra.nn.features import FeatureMaps, FeatureSpec


def test_mask_rcnn_head_returns_masks_with_expected_shape():
    in_spec = FeatureSpec(channels={"roi": 64})
    cfg = MaskRCNNHeadConfig(conv_dims=(32, 32), upsample_dim=16, num_classes=5)
    head = MaskRCNNHead(in_spec=in_spec, cfg=cfg)

    features = FeatureMaps(feature_maps={"roi": torch.randn(3, 64, 7, 7)})
    out = head(features)

    assert out.masks.shape == (3, 5, 14, 14)
    assert out.logits is None


def test_mask_rcnn_head_can_be_class_agnostic():
    in_spec = FeatureSpec(channels={"roi": 32})
    cfg = MaskRCNNHeadConfig(conv_dims=(16,), upsample_dim=8, class_agnostic=True)
    head = MaskRCNNHead(in_spec=in_spec, cfg=cfg)

    features = FeatureMaps(feature_maps={"roi": torch.randn(2, 32, 7, 7)})
    out = head(features)

    assert out.masks.shape == (2, 1, 14, 14)


def test_mask_rcnn_head_factory_returns_head_instance():
    in_spec = FeatureSpec(channels={"roi": 16})
    cfg = MaskRCNNHeadConfig(conv_dims=(8,), upsample_dim=8, num_classes=3)

    head = mask_rcnn_head(in_spec=in_spec, cfg=cfg)

    assert isinstance(head, MaskRCNNHead)
