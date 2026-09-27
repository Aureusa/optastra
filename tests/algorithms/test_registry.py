import pytest

import optastra  # noqa: F401  -- registers every built-in component
from optastra.algorithms.base import Algorithm
from optastra.core.factory import FACTORIES
from optastra.tasks.base import Task

_registry = Algorithm._registry


def test_algorithms_have_their_own_registry_family():
    assert _registry is not Task._registry
    assert _registry.family == "algorithm"
    assert FACTORIES["algorithm"] is Algorithm
    assert FACTORIES["task"] is Task


def test_algorithm_and_task_registries_do_not_leak_into_each_other():
    assert {"byol", "simclr"} <= set(Algorithm.list_all())
    assert "classification_task" not in Algorithm.list_all()
    assert not {"byol", "simclr"} & set(Task.list_all())

    with pytest.raises(ValueError, match="algorithm 'classification_task' is not registered"):
        Algorithm.create("classification_task")
    with pytest.raises(ValueError, match="task 'byol' is not registered"):
        Task.create("byol")


def test_algorithms_register_through_their_own_registry():
    @Algorithm.register
    def UnitTestAlgorithmRegistryFn():
        return "ok"

    assert "UnitTestAlgorithmRegistryFn" in _registry.list_component()
    assert _registry.get_entrypoint("UnitTestAlgorithmRegistryFn") is UnitTestAlgorithmRegistryFn
    assert _registry.get_module("UnitTestAlgorithmRegistryFn") == __name__.split(".")[-1]
    assert _registry.is_registered("UnitTestAlgorithmRegistryFn") is True
    assert not Task._registry.is_registered("UnitTestAlgorithmRegistryFn")


def test_algorithms_support_default_config_and_duplicate_rejection():
    cfg = {"temperature": 0.2}

    @Algorithm.register(config=cfg)
    def UnitTestAlgorithmWithConfig():
        return "ok"

    assert _registry.get_default_config("UnitTestAlgorithmWithConfig") == cfg

    with pytest.raises(ValueError, match="algorithm UnitTestAlgorithmWithConfig already registered"):
        @Algorithm.register(config=cfg)
        def UnitTestAlgorithmWithConfig():
            return "dup"


def test_algorithm_registry_missing_entries_raise():
    with pytest.raises(ValueError, match="algorithm MissingAlgorithm is not registered"):
        _registry.get_entrypoint("MissingAlgorithm")

    with pytest.raises(ValueError, match="algorithm MissingAlgorithm is not registered"):
        _registry.get_module("MissingAlgorithm")

    with pytest.raises(ValueError, match="algorithm MissingAlgorithm is not registered"):
        _registry.get_default_config("MissingAlgorithm")


def test_simclr_and_byol_create_their_algorithm_classes():
    from optastra.algorithms import BYOLTask, SimCLRTask

    assert isinstance(Algorithm.create("simclr"), SimCLRTask)
    assert isinstance(Algorithm.create("byol"), BYOLTask)
    with pytest.warns(DeprecationWarning, match="use 'simclr'"):
        assert isinstance(Algorithm.create("simclr_no_momentum"), SimCLRTask)


def test_algorithms_declare_multiview_collate():
    for name in ("simclr", "byol"):
        assert Algorithm.create(name).collate == "multiview"
