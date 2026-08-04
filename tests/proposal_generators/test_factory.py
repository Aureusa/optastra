from optastra.proposal_generators import ProposalGenerator
from optastra.nn.features import FeatureSpec


def test_proposal_generator_config_returns_rpn_defaults():
    cfg = ProposalGenerator.config("rpn")

    assert cfg.num_anchors == 3
    assert cfg.box_dim == 4


def test_proposal_generator_create_builds_rpn_and_sets_out_spec():
    model = ProposalGenerator.create(
        "rpn",
        in_spec=FeatureSpec(channels={"P3": 64, "P4": 64}, strides={"P3": 8, "P4": 16}),
        num_anchors=7,
    )

    assert model.out_spec.channels["P3_objectness"] == 7
    assert model.out_spec.channels["P4_deltas"] == 28
