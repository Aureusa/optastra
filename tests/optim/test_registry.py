import pytest

from optastra.optim._registry import (
    check_optimizer_registered,
    check_scheduler_registered,
    get_optimizer_default_config,
    get_optimizer_entrypoint,
    get_optimizer_module,
    get_scheduler_default_config,
    get_scheduler_entrypoint,
    list_optimizers,
    list_schedulers,
    register_optimizer,
    register_scheduler,
)


def test_optimizer_registry_registers_and_lists_components():
    @register_optimizer
    def UnitTestOptimizerRegistryFn(param_groups, cfg):
        return (param_groups, cfg)

    assert "UnitTestOptimizerRegistryFn" in list_optimizers()
    assert get_optimizer_entrypoint("UnitTestOptimizerRegistryFn") is UnitTestOptimizerRegistryFn
    assert get_optimizer_module("UnitTestOptimizerRegistryFn") == __name__.split(".")[-1]
    assert check_optimizer_registered("UnitTestOptimizerRegistryFn") is True


def test_optimizer_registry_default_config_and_duplicate_rejection():
    cfg = {"lr": 0.1}

    @register_optimizer(config=cfg)
    def UnitTestOptimizerWithConfig(param_groups, cfg):
        return (param_groups, cfg)

    assert get_optimizer_default_config("UnitTestOptimizerWithConfig") == cfg

    with pytest.raises(ValueError, match="optimizer UnitTestOptimizerWithConfig already registered"):
        @register_optimizer(config=cfg)
        def UnitTestOptimizerWithConfig(param_groups, cfg):
            return (param_groups, cfg)


def test_scheduler_registry_registers_and_lists_components():
    @register_scheduler
    def UnitTestSchedulerRegistryFn(optimizer, cfg):
        return (optimizer, cfg)

    assert "UnitTestSchedulerRegistryFn" in list_schedulers()
    assert get_scheduler_entrypoint("UnitTestSchedulerRegistryFn") is UnitTestSchedulerRegistryFn
    assert check_scheduler_registered("UnitTestSchedulerRegistryFn") is True


def test_scheduler_registry_default_config_and_duplicate_rejection():
    cfg = {"total_steps": 10}

    @register_scheduler(config=cfg)
    def UnitTestSchedulerWithConfig(optimizer, cfg):
        return (optimizer, cfg)

    assert get_scheduler_default_config("UnitTestSchedulerWithConfig") == cfg

    with pytest.raises(ValueError, match="scheduler UnitTestSchedulerWithConfig already registered"):
        @register_scheduler(config=cfg)
        def UnitTestSchedulerWithConfig(optimizer, cfg):
            return (optimizer, cfg)
