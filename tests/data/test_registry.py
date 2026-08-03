import pytest

from optastra.data._registry import (
    check_collate_registered,
    get_collate_default_config,
    get_collate_entrypoint,
    get_collate_module,
    list_collates,
    register_collate,
)


def test_collate_registry_registers_and_retrieves_entrypoint():
    @register_collate
    def UnitTestCollateRegistryFn(batch):
        return batch

    assert "UnitTestCollateRegistryFn" in list_collates()
    assert get_collate_entrypoint("UnitTestCollateRegistryFn") is UnitTestCollateRegistryFn
    assert get_collate_module("UnitTestCollateRegistryFn") == __name__.split(".")[-1]
    assert check_collate_registered("UnitTestCollateRegistryFn") is True


def test_collate_registry_stores_default_config_and_detects_duplicates():
    cfg = {"some": "value"}

    @register_collate(config=cfg)
    def UnitTestCollateWithConfig(batch):
        return batch

    assert get_collate_default_config("UnitTestCollateWithConfig") == cfg

    with pytest.raises(ValueError, match="collate UnitTestCollateWithConfig already registered"):
        @register_collate(config=cfg)
        def UnitTestCollateWithConfig(batch):
            return batch


def test_collate_registry_raises_for_missing_entries():
    with pytest.raises(ValueError, match="collate MissingCollate is not registered"):
        get_collate_entrypoint("MissingCollate")

    with pytest.raises(ValueError, match="collate MissingCollate is not registered"):
        get_collate_module("MissingCollate")

    with pytest.raises(ValueError, match="collate MissingCollate is not registered"):
        get_collate_default_config("MissingCollate")
