import pytest

from optastra.algorithms._registry import (
    check_algorithm_registered,
    get_algorithm_default_config,
    get_algorithm_entrypoint,
    get_algorithm_module,
    list_algorithms,
    register_algorithm,
)


def test_algorithm_registry_registers_and_lists_components():
    @register_algorithm
    def UnitTestAlgorithmRegistryFn():
        return "ok"

    assert "UnitTestAlgorithmRegistryFn" in list_algorithms()
    assert get_algorithm_entrypoint("UnitTestAlgorithmRegistryFn") is UnitTestAlgorithmRegistryFn
    assert get_algorithm_module("UnitTestAlgorithmRegistryFn") == __name__.split(".")[-1]
    assert check_algorithm_registered("UnitTestAlgorithmRegistryFn") is True


def test_algorithm_registry_default_config_and_duplicate_rejection():
    cfg = {"temperature": 0.2}

    @register_algorithm(config=cfg)
    def UnitTestAlgorithmWithConfig():
        return "ok"

    assert get_algorithm_default_config("UnitTestAlgorithmWithConfig") == cfg

    with pytest.raises(ValueError, match="algorithm UnitTestAlgorithmWithConfig already registered"):
        @register_algorithm(config=cfg)
        def UnitTestAlgorithmWithConfig():
            return "dup"


def test_algorithm_registry_missing_entries_raise():
    with pytest.raises(ValueError, match="algorithm MissingAlgorithm is not registered"):
        get_algorithm_entrypoint("MissingAlgorithm")

    with pytest.raises(ValueError, match="algorithm MissingAlgorithm is not registered"):
        get_algorithm_module("MissingAlgorithm")

    with pytest.raises(ValueError, match="algorithm MissingAlgorithm is not registered"):
        get_algorithm_default_config("MissingAlgorithm")
