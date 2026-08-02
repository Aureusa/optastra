from typing import Any, Callable, List, Optional

from optastra.registry import ComponentRegistry


_registry = ComponentRegistry("head")


def register_head(
    fn: Optional[Callable[..., Any]] = None,
    *,
    config: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
    """Decorator for registering heads.

    Supports both usages:
    - @register_head
    - @register_head(config=...)
    """

    def decorator(inner_fn: Callable[..., Any]) -> Callable[..., Any]:
        return _registry.register(inner_fn, default_config=config)

    if fn is not None:
        return decorator(fn)
    return decorator


def list_heads(module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
    """List all registered head names, optionally filtered by module and/or a wildcard pattern."""
    return _registry.list_component(module=module, filter=filter)


def get_head_entrypoint(name: str) -> Callable[..., Any]:
    """Get the entrypoint function for a registered head by name."""
    return _registry.get_entrypoint(name)


def get_head_module(name: str) -> str:
    """Get the module name for a registered head by name."""
    return _registry.get_module(name)

def get_head_default_config(name: str) -> Any:
    """Get the default configuration for a registered head by name."""
    return _registry.get_default_config(name)

def check_head_registered(name: str) -> bool:
    """Check if a head is registered by name."""
    return _registry.is_registered(name)
