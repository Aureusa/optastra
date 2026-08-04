import pytest

from optastra.heads._registry import _registry, register_head


def test_head_registry_registers_and_retrieves_entrypoint():
    @register_head
    def UnitTestHeadRegistryFn(*args, **kwargs):
        return args, kwargs

    assert "UnitTestHeadRegistryFn" in _registry.list_component()
    assert _registry.get_entrypoint("UnitTestHeadRegistryFn") is UnitTestHeadRegistryFn
    assert _registry.get_module("UnitTestHeadRegistryFn") == __name__.split(".")[-1]
    assert _registry.is_registered("UnitTestHeadRegistryFn") is True


def test_head_registry_default_config_and_duplicate_rejection():
    cfg = {"num_classes": 11}

    @register_head(config=cfg)
    def UnitTestHeadWithConfig(*args, **kwargs):
        return args, kwargs

    assert _registry.get_default_config("UnitTestHeadWithConfig") == cfg

    with pytest.raises(ValueError, match="head UnitTestHeadWithConfig already registered"):
        @register_head(config=cfg)
        def UnitTestHeadWithConfig(*args, **kwargs):
            return None


def test_head_registry_raises_for_missing_entries():
    with pytest.raises(ValueError, match="head MissingHead is not registered"):
        _registry.get_entrypoint("MissingHead")

    with pytest.raises(ValueError, match="head MissingHead is not registered"):
        _registry.get_module("MissingHead")

    with pytest.raises(ValueError, match="head MissingHead is not registered"):
        _registry.get_default_config("MissingHead")
