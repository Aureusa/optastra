import pytest

from optastra.backbones.base import Backbone

_registry = Backbone._registry


def test_backbone_registry_registers_and_retrieves_entrypoint():
    @Backbone.register
    def UnitTestBackboneRegistryFn():
        return "ok"

    assert "UnitTestBackboneRegistryFn" in _registry.list_component()
    assert _registry.get_entrypoint("UnitTestBackboneRegistryFn") is UnitTestBackboneRegistryFn
    assert _registry.get_module("UnitTestBackboneRegistryFn") == __name__.split(".")[-1]
    assert _registry.is_registered("UnitTestBackboneRegistryFn") is True


def test_backbone_registry_default_config_and_duplicate_rejection():
    cfg = {"depth": 18}

    @Backbone.register(config=cfg)
    def UnitTestBackboneWithConfig():
        return "ok"

    assert _registry.get_default_config("UnitTestBackboneWithConfig") == cfg

    with pytest.raises(ValueError, match="backbone UnitTestBackboneWithConfig already registered"):
        @Backbone.register(config=cfg)
        def UnitTestBackboneWithConfig():
            return "dup"


def test_backbone_registry_filtering_and_missing_entries():
    @Backbone.register
    def UnitTestBackboneForFilter():
        return "ok"

    assert "UnitTestBackboneForFilter" in _registry.list_component(filter="UnitTestBackbone*")
    assert _registry.list_component(module="DoesNotExist") == []

    with pytest.raises(ValueError, match="backbone MissingBackbone is not registered"):
        _registry.get_entrypoint("MissingBackbone")

    with pytest.raises(ValueError, match="backbone MissingBackbone is not registered"):
        _registry.get_module("MissingBackbone")

    with pytest.raises(ValueError, match="backbone MissingBackbone is not registered"):
        _registry.get_default_config("MissingBackbone")