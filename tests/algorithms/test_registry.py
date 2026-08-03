import pytest

from optastra.tasks._registry import (
    check_task_registered,
    get_task_default_config,
    get_task_entrypoint,
    get_task_module,
    list_tasks,
    register_task,
)


def test_algorithms_register_through_shared_task_registry():
    @register_task
    def UnitTestAlgorithmRegistryFn():
        return "ok"

    assert "UnitTestAlgorithmRegistryFn" in list_tasks()
    assert get_task_entrypoint("UnitTestAlgorithmRegistryFn") is UnitTestAlgorithmRegistryFn
    assert get_task_module("UnitTestAlgorithmRegistryFn") == __name__.split(".")[-1]
    assert check_task_registered("UnitTestAlgorithmRegistryFn") is True


def test_algorithms_support_default_config_and_duplicate_rejection_via_task_registry():
    cfg = {"temperature": 0.2}

    @register_task(config=cfg)
    def UnitTestAlgorithmWithConfig():
        return "ok"

    assert get_task_default_config("UnitTestAlgorithmWithConfig") == cfg

    with pytest.raises(ValueError, match="task UnitTestAlgorithmWithConfig already registered"):
        @register_task(config=cfg)
        def UnitTestAlgorithmWithConfig():
            return "dup"


def test_algorithm_task_registry_missing_entries_raise():
    with pytest.raises(ValueError, match="task MissingAlgorithm is not registered"):
        get_task_entrypoint("MissingAlgorithm")

    with pytest.raises(ValueError, match="task MissingAlgorithm is not registered"):
        get_task_module("MissingAlgorithm")

    with pytest.raises(ValueError, match="task MissingAlgorithm is not registered"):
        get_task_default_config("MissingAlgorithm")


def test_simclr_algorithm_is_registered_as_task_with_default_config():
    # Import for registration side effect.
    import optastra.algorithms.simclr.routine  # noqa: F401

    assert check_task_registered("simclr_no_momentum") is True
    entrypoint = get_task_entrypoint("simclr_no_momentum")
    cfg = get_task_default_config("simclr_no_momentum")

    created = entrypoint(cfg)

    assert created.__class__.__name__ == "SimCLRTask"
