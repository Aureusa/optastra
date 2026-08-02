import pytest

from optastra.necks import Neck
from optastra.nn.features import FeatureSpec


def _fpn_in_spec() -> FeatureSpec:
    return FeatureSpec(
        channels={"C2": 256, "C3": 512, "C4": 1024, "C5": 2048},
        strides={"C2": 4, "C3": 8, "C4": 16, "C5": 32},
    )


def test_all_registered_necks_initialization():
    necks = Neck.list_necks()

    for neck_name in necks:
        model = Neck.create(neck_name, _fpn_in_spec())
        assert model is not None


def test_failure_on_unknown_neck():
    with pytest.raises(ValueError):
        Neck.create("unknown_neck", _fpn_in_spec())


def test_neck_create_requires_in_spec_when_backbone_is_missing():
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        Neck.create("fpn")

def test_neck_create_requires_with_in_spec_not_feature_spec():
    with pytest.raises(TypeError, match="must be a FeatureSpec"):
        Neck.create("fpn", {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048})


def test_neck_create_requires_in_spec_with_channels_and_strides():
    with pytest.raises(ValueError, match="missing"):
        Neck.create("fpn", FeatureSpec(channels={"C2": 256}))

def test_neck_create_with_unknown_override_raises_type_error():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        Neck.create("fpn", _fpn_in_spec(), not_a_real_field=123)
