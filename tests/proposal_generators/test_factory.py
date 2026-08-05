from optastra.proposal_generators import ProposalGenerator
from optastra.nn.features import FeatureSpec


def test_proposal_generator_config_returns_rpn_defaults():
    cfg = ProposalGenerator.get_default_config("rpn")

    assert cfg.num_anchors is None
    assert cfg.box_dim == 4
    assert len(cfg.anchor_scales) * len(cfg.aspect_ratios) == 3


def test_proposal_generator_create_builds_rpn_and_sets_out_spec():
    model = ProposalGenerator.create(
        "rpn",
        in_spec=FeatureSpec(channels={"P3": 64, "P4": 64}, strides={"P3": 8, "P4": 16}),
        num_anchors=7,
        anchor_scales=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0),
        aspect_ratios=(1.0,),
    )

    assert model.out_spec.channels["P3_objectness"] == 7
    assert model.out_spec.channels["P4_deltas"] == 28
