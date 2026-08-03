from typing import Any, Callable, List, Optional

from optastra.registry import ComponentRegistry


_registry = ComponentRegistry("optimizer")
_scheduler_registry = ComponentRegistry("scheduler")


def register_optimizer(
    fn: Optional[Callable[..., Any]] = None,
    *,
    config: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    """Decorator for registering optimizers.

    Supports both usages:
    - @register_optimizer
    - @register_optimizer(config=...)
    """

    def decorator(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
        return _registry.register(inner_fn, default_config=config)

    if fn is not None:
        return decorator(fn)
    return decorator


def list_optimizers(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    """List all registered optimizer names, optionally filtered by module and/or a wildcard pattern."""
    return _registry.list_component(module=module, filter=filter)


def get_optimizer_entrypoint(name: str) -> Callable[..., Any]:
    """Get the entrypoint function for a registered optimizer by name."""
    return _registry.get_entrypoint(name)

def get_optimizer_module(name: str) -> str:
    """Get the module name for a registered optimizer by name."""
    return _registry.get_module(name)

def get_optimizer_default_config(name: str) -> Any:
    """Get the default configuration for a registered optimizer by name."""
    return _registry.get_default_config(name)

def check_optimizer_registered(name: str) -> bool:
    """Check if a optimizer is registered by name."""
    return _registry.is_registered(name)


##########################################
########### Scheduler Registry ###########
##########################################

def register_scheduler(
    fn: Optional[Callable[..., Any]] = None,
    *,
    config: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    def decorator(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
        return _scheduler_registry.register(inner_fn, default_config=config)

    if fn is not None:
        return decorator(fn)
    return decorator

def list_schedulers(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    return _scheduler_registry.list_component(module=module, filter=filter)

def get_scheduler_entrypoint(name: str) -> Callable[..., Any]:
    return _scheduler_registry.get_entrypoint(name)

def get_scheduler_default_config(name: str) -> Any:
    return _scheduler_registry.get_default_config(name)

def check_scheduler_registered(name: str) -> bool:
    return _scheduler_registry.is_registered(name)
