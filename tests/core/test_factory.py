from dataclasses import dataclass, field

import pytest

from optastra import get_component_parameters
from optastra.backbones import Backbone
from optastra.core.factory import Factory
from optastra.core.registry import FamilyRegistry


class _Widget(Factory["_Widget"]):
    _registry = FamilyRegistry("unit_test_widget")


@dataclass
class _WidgetConfig:
    sizes: list[int] = field(default_factory=lambda: [1, 2])
    nested: dict = field(default_factory=lambda: {"a": [1]})


@_Widget.register(config=_WidgetConfig())
def widget(cfg):
    return cfg


@_Widget.register
def configless_widget():
    return "built"


for _name, _cfg in {"widget_small": _WidgetConfig(sizes=[1]), "widget_big": _WidgetConfig(sizes=[8])}.items():
    _Widget.register(widget, config=_cfg, name=_name)


def test_register_with_name_in_a_loop():
    assert _Widget.list_all(filter="widget_") == ["widget_big", "widget_small"]
    assert _Widget.create("widget_big").sizes == [8]


def test_created_config_is_a_deep_copy_of_the_default():
    cfg = _Widget.create("widget")
    cfg.sizes.append(3)
    cfg.nested["a"].append(2)

    fresh = _Widget.create("widget")
    assert fresh.sizes == [1, 2]
    assert fresh.nested == {"a": [1]}


def test_get_default_config_returns_a_copy():
    _Widget.get_default_config("widget").sizes.clear()
    assert _Widget.get_default_config("widget").sizes == [1, 2]


def test_configless_component_builds_without_cfg_and_rejects_overrides():
    assert _Widget.create("configless_widget") == "built"
    assert _Widget.get_default_config("configless_widget") is None
    assert get_component_parameters("configless_widget", family="unit_test_widget") == {}
    with pytest.raises(TypeError, match="takes no overrides"):
        _Widget.create("configless_widget", size=3)


def test_describe_configless(capsys):
    _Widget.describe("configless_widget")
    assert "(no config)" in capsys.readouterr().out


def test_backbone_factory_functions_are_not_injected_into_package_exports():
    import optastra.backbones as backbones
    import optastra.backbones.resnet as resnet

    assert "resnet50" not in resnet.__all__
    assert "ResNetConfig" in backbones.__all__
    assert Backbone.create("resnet18") is not None
