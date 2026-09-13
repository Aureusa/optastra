import pytest

from optastra.optim.base import Optimizer
from optastra.optim.scheduler_base import Scheduler

_registry = Optimizer._registry
_scheduler_registry = Scheduler._registry


def test_optimizer_registry_registers_and_lists_components():
    @Optimizer.register
    def UnitTestOptimizerRegistryFn(param_groups, cfg):
        return (param_groups, cfg)

    assert "UnitTestOptimizerRegistryFn" in _registry.list_component()
    assert _registry.get_entrypoint("UnitTestOptimizerRegistryFn") is UnitTestOptimizerRegistryFn
    assert _registry.get_module("UnitTestOptimizerRegistryFn") == __name__.split(".")[-1]
    assert _registry.is_registered("UnitTestOptimizerRegistryFn") is True


def test_optimizer_registry_default_config_and_duplicate_rejection():
    cfg = {"lr": 0.1}

    @Optimizer.register(config=cfg)
    def UnitTestOptimizerWithConfig(param_groups, cfg):
        return (param_groups, cfg)

    assert _registry.get_default_config("UnitTestOptimizerWithConfig") == cfg

    with pytest.raises(ValueError, match="optimizer UnitTestOptimizerWithConfig already registered"):
        @Optimizer.register(config=cfg)
        def UnitTestOptimizerWithConfig(param_groups, cfg):
            return (param_groups, cfg)


def test_scheduler_registry_registers_and_lists_components():
    @Scheduler.register
    def UnitTestSchedulerRegistryFn(optimizer, cfg):
        return (optimizer, cfg)

    assert "UnitTestSchedulerRegistryFn" in _scheduler_registry.list_component()
    assert _scheduler_registry.get_entrypoint("UnitTestSchedulerRegistryFn") is UnitTestSchedulerRegistryFn
    assert _scheduler_registry.is_registered("UnitTestSchedulerRegistryFn") is True


def test_scheduler_registry_default_config_and_duplicate_rejection():
    cfg = {"total_steps": 10}

    @Scheduler.register(config=cfg)
    def UnitTestSchedulerWithConfig(optimizer, cfg):
        return (optimizer, cfg)

    assert _scheduler_registry.get_default_config("UnitTestSchedulerWithConfig") == cfg

    with pytest.raises(ValueError, match="scheduler UnitTestSchedulerWithConfig already registered"):
        @Scheduler.register(config=cfg)
        def UnitTestSchedulerWithConfig(optimizer, cfg):
            return (optimizer, cfg)
