import pytest

from optastra.heads._registry import (
    check_head_registered,
    get_head_default_config,
    get_head_entrypoint,
    get_head_module,
    list_heads,
    register_head,
)


def test_head_registry_registers_and_retrieves_entrypoint():
    @register_head
    def UnitTestHeadRegistryFn(*args, **kwargs):
        return args, kwargs

    assert "UnitTestHeadRegistryFn" in list_heads()
    assert get_head_entrypoint("UnitTestHeadRegistryFn") is UnitTestHeadRegistryFn
    assert get_head_module("UnitTestHeadRegistryFn") == __name__.split(".")[-1]
    assert check_head_registered("UnitTestHeadRegistryFn") is True


def test_head_registry_default_config_and_duplicate_rejection():
    cfg = {"num_classes": 11}

    @register_head(config=cfg)
    def UnitTestHeadWithConfig(*args, **kwargs):
        return args, kwargs

    assert get_head_default_config("UnitTestHeadWithConfig") == cfg

    with pytest.raises(ValueError, match="head UnitTestHeadWithConfig already registered"):
        @register_head(config=cfg)
        def UnitTestHeadWithConfig(*args, **kwargs):
            return None


def test_head_registry_raises_for_missing_entries():
    with pytest.raises(ValueError, match="head MissingHead is not registered"):
        get_head_entrypoint("MissingHead")

    with pytest.raises(ValueError, match="head MissingHead is not registered"):
        get_head_module("MissingHead")

    with pytest.raises(ValueError, match="head MissingHead is not registered"):
        get_head_default_config("MissingHead")
