import pytest

from optastra.core.registry import ComponentRegistry


def test_component_registry_registers_and_lists_components():
    registry = ComponentRegistry("component")

    @registry.register
    def DummyComponent():
        return "ok"

    assert "DummyComponent" in registry.list_component()
    assert registry.get_entrypoint("DummyComponent") is DummyComponent
    assert registry.get_module("DummyComponent") == __name__.split('.')[-1]


def test_component_registry_rejects_duplicate_registration():
    registry = ComponentRegistry("component")

    @registry.register
    def DuplicateComponent():
        return "first"

    with pytest.raises(ValueError, match="component DuplicateComponent already registered"):
        @registry.register
        def DuplicateComponent():
            return "second"
