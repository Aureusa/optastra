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
