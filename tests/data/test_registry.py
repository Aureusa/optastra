import pytest

from optastra.data.collate import CollateFn, register_collate


def test_collate_registry_registers_and_retrieves_entrypoint():
    @register_collate
    def UnitTestCollateRegistryFn(batch):
        return batch

    registry = CollateFn._registry
    assert "UnitTestCollateRegistryFn" in registry.list_component()
    assert registry.get_entrypoint("UnitTestCollateRegistryFn") is UnitTestCollateRegistryFn
    assert registry.get_module("UnitTestCollateRegistryFn") == __name__.split(".")[-1]
    assert registry.is_registered("UnitTestCollateRegistryFn") is True


def test_collate_registry_stores_default_config_and_detects_duplicates():
    cfg = {"some": "value"}

    @register_collate(config=cfg)
    def UnitTestCollateWithConfig(batch):
        return batch

    assert CollateFn._registry.get_default_config("UnitTestCollateWithConfig") == cfg

    with pytest.raises(ValueError, match="collate UnitTestCollateWithConfig already registered"):
        @register_collate(config=cfg)
        def UnitTestCollateWithConfig(batch):
            return batch


def test_collate_registry_raises_for_missing_entries():
    with pytest.raises(ValueError, match="collate MissingCollate is not registered"):
        CollateFn._registry.get_entrypoint("MissingCollate")

    with pytest.raises(ValueError, match="collate MissingCollate is not registered"):
        CollateFn._registry.get_module("MissingCollate")

    with pytest.raises(ValueError, match="collate MissingCollate is not registered"):
        CollateFn._registry.get_default_config("MissingCollate")
