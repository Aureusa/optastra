import pytest

from optastra.tasks._registry import (
    check_task_registered,
    get_task_default_config,
    get_task_entrypoint,
    get_task_module,
    list_tasks,
    register_task,
)


def test_task_registry_registers_and_retrieves_entrypoint():
    @register_task
    def UnitTestTaskRegistryFn():
        return "ok"

    assert "UnitTestTaskRegistryFn" in list_tasks()
    assert get_task_entrypoint("UnitTestTaskRegistryFn") is UnitTestTaskRegistryFn
    assert get_task_module("UnitTestTaskRegistryFn") == __name__.split(".")[-1]
    assert check_task_registered("UnitTestTaskRegistryFn") is True


def test_task_registry_default_config_and_duplicate_rejection():
    cfg = {"label_smoothing": 0.1}

    @register_task(config=cfg)
    def UnitTestTaskWithConfig():
        return "ok"

    assert get_task_default_config("UnitTestTaskWithConfig") == cfg

    with pytest.raises(ValueError, match="task UnitTestTaskWithConfig already registered"):
        @register_task(config=cfg)
        def UnitTestTaskWithConfig():
            return "dup"


def test_task_registry_raises_for_missing_entries():
    with pytest.raises(ValueError, match="task MissingTask is not registered"):
        get_task_entrypoint("MissingTask")

    with pytest.raises(ValueError, match="task MissingTask is not registered"):
        get_task_module("MissingTask")

    with pytest.raises(ValueError, match="task MissingTask is not registered"):
        get_task_default_config("MissingTask")
