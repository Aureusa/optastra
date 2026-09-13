import pytest

from optastra.tasks.base import Task

_registry = Task._registry


def test_task_registry_registers_and_retrieves_entrypoint():
    @Task.register
    def UnitTestTaskRegistryFn():
        return "ok"

    assert "UnitTestTaskRegistryFn" in _registry.list_component()
    assert _registry.get_entrypoint("UnitTestTaskRegistryFn") is UnitTestTaskRegistryFn
    assert _registry.get_module("UnitTestTaskRegistryFn") == __name__.split(".")[-1]
    assert _registry.is_registered("UnitTestTaskRegistryFn") is True


def test_task_registry_default_config_and_duplicate_rejection():
    cfg = {"label_smoothing": 0.1}

    @Task.register(config=cfg)
    def UnitTestTaskWithConfig():
        return "ok"

    assert _registry.get_default_config("UnitTestTaskWithConfig") == cfg

    with pytest.raises(ValueError, match="task UnitTestTaskWithConfig already registered"):
        @Task.register(config=cfg)
        def UnitTestTaskWithConfig():
            return "dup"


def test_task_registry_raises_for_missing_entries():
    with pytest.raises(ValueError, match="task MissingTask is not registered"):
        _registry.get_entrypoint("MissingTask")

    with pytest.raises(ValueError, match="task MissingTask is not registered"):
        _registry.get_module("MissingTask")

    with pytest.raises(ValueError, match="task MissingTask is not registered"):
        _registry.get_default_config("MissingTask")
