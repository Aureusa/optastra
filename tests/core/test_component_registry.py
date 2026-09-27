import pytest

from optastra.core.registry import FamilyRegistry


def test_component_registry_registers_and_lists_components():
    registry = FamilyRegistry("component")

    @registry.register
    def DummyComponent():
        return "ok"

    assert "DummyComponent" in registry.list_component()
    assert registry.get_entrypoint("DummyComponent") is DummyComponent
    assert registry.get_module("DummyComponent") == __name__.split('.')[-1]


def test_component_registry_rejects_duplicate_registration():
    registry = FamilyRegistry("component")

    @registry.register
    def DuplicateComponent():
        return "first"

    with pytest.raises(ValueError, match="component DuplicateComponent already registered"):
        @registry.register
        def DuplicateComponent():
            return "second"


def test_component_registry_filter_uses_regex_search():
    registry = FamilyRegistry("component")

    @registry.register
    def mask_rcnn_1():
        return "m1"

    @registry.register
    def mask_rcnn_2():
        return "m2"

    @registry.register
    def faster_rcnn_r50():
        return "f"

    assert registry.list_component(filter="mask_rcnn") == ["mask_rcnn_1", "mask_rcnn_2"]
    assert registry.list_component(filter="rcnn") == ["faster_rcnn_r50", "mask_rcnn_1", "mask_rcnn_2"]
    assert registry.list_component(filter="mask_rcnn_[12]") == ["mask_rcnn_1", "mask_rcnn_2"]


def test_component_registry_accepts_explicit_name_for_loop_registration():
    registry = FamilyRegistry("component")

    def build(cfg):
        return cfg

    for variant, cfg in {"small": {"width": 1}, "large": {"width": 2}}.items():
        registry.register(build, default_config=cfg, name=variant)

    assert registry.list_component() == ["large", "small"]
    assert registry.get_entrypoint("small") is build
    assert registry.get_default_config("large") == {"width": 2}

    with pytest.raises(ValueError, match="component small already registered"):
        registry.register(build, name="small")


def test_component_registry_does_not_touch_module_all():
    import sys

    module = sys.modules[__name__]
    had_all = hasattr(module, "__all__")
    registry = FamilyRegistry("component")

    @registry.register
    def ComponentNotExported():
        return "ok"

    assert hasattr(module, "__all__") == had_all


def test_component_registry_configless_entry_has_no_default_config():
    registry = FamilyRegistry("component")

    @registry.register
    def Configless():
        return "ok"

    assert registry.get_default_config("Configless") is None
