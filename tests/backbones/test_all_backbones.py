import pytest
from optastra.backbones import Backbone


def test_all_registered_backbones_initialization():
    backbones = Backbone.list_backbones()

    for backbone_name in backbones:
        model = Backbone.create(backbone_name)

def test_failure_on_unknown_backbone():
    with pytest.raises(ValueError):
        Backbone.create("unknown_backbone")
