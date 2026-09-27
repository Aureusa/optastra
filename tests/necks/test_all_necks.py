import pytest

from optastra.necks import Neck
from optastra.nn.features import FeatureSpec


def _fpn_in_spec() -> FeatureSpec:
    return FeatureSpec(
        channels={"C2": 256, "C3": 512, "C4": 1024, "C5": 2048},
        strides={"C2": 4, "C3": 8, "C4": 16, "C5": 32},
    )


def _vit_like_in_spec() -> FeatureSpec:
    # what a ViT exposes: a token view (embed_dim) and a single stride-16 map
    return FeatureSpec(channels={"C4": 192}, strides={"C4": 16}, embed_dim=192, num_tokens=196)


@pytest.mark.parametrize("neck_name", Neck.list_all())
def test_all_registered_necks_initialization(neck_name):
    if neck_name == "token_pool":
        with pytest.raises(ValueError, match="embed_dim"):
            Neck.create(neck_name, _fpn_in_spec())   # CNN specs carry no tokens
    else:
        assert Neck.create(neck_name, _fpn_in_spec()) is not None
    assert Neck.create(neck_name, _vit_like_in_spec()) is not None


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
