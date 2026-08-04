import pytest
import torch

from optastra.nn.features import FeatureMaps, FeatureSpec
from optastra.proposal_generators import ProposalGenerator
from optastra.proposal_generators.rpn import RPN, RPNConfig


def test_rpn_forward_single_feature_map_shapes():
    in_spec = FeatureSpec(channels={"P3": 64}, strides={"P3": 8})
    rpn = RPN(in_spec=in_spec, cfg=RPNConfig(num_anchors=3))
    x = FeatureMaps(feature_maps={"P3": torch.randn(2, 64, 32, 32)})

    out = rpn(x)

    assert out.feature_maps["P3_objectness"].shape == (2, 3, 32, 32)
    assert out.feature_maps["P3_deltas"].shape == (2, 12, 32, 32)


def test_rpn_forward_multi_level_shapes():
    in_spec = FeatureSpec(
        channels={"P3": 32, "P4": 32},
        strides={"P3": 8, "P4": 16},
    )
    rpn = RPN(
        in_spec=in_spec,
        cfg=RPNConfig(num_anchors=5, conv_dims=(64, 64), in_features=("P3", "P4")),
    )
    feats = FeatureMaps(
        feature_maps={
            "P3": torch.randn(2, 32, 64, 64),
            "P4": torch.randn(2, 32, 32, 32),
        }
    )

    out = rpn(feats)

    assert out.feature_maps["P3_objectness"].shape == (2, 5, 64, 64)
    assert out.feature_maps["P3_deltas"].shape == (2, 20, 64, 64)
    assert out.feature_maps["P4_objectness"].shape == (2, 5, 32, 32)
    assert out.feature_maps["P4_deltas"].shape == (2, 20, 32, 32)


def test_rpn_rejects_invalid_conv_dims():
    with pytest.raises(ValueError, match="must be > 0"):
        RPN(
            in_spec=FeatureSpec(channels={"P3": 16}, strides={"P3": 8}),
            cfg=RPNConfig(num_anchors=3, conv_dims=(0,)),
        )


def test_rpn_factory_builds_registered_module():
    model = ProposalGenerator.create(
        "rpn",
        in_spec=FeatureSpec(channels={"P3": 64}, strides={"P3": 8}),
        num_anchors=9,
    )

    assert isinstance(model, RPN)
    assert model.out_spec.channels["P3_objectness"] == 9
