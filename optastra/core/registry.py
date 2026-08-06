from dataclasses import dataclass
import re
import sys
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, TypeVar


T = TypeVar("T", bound=Callable[..., Any])


@dataclass
class RegistryEntry:
    name: str
    entrypoint: Callable
    default_config: Any
    module: str


class FamilyRegistry:

    def __init__(
        self,
        family: str,
    ):
        self.family = family
        self._components: Dict[str, RegistryEntry] = {}

    def register(self, fn: T, *, default_config: Optional[Any] = None) -> T:
        mod = sys.modules[fn.__module__]
        module_name = fn.__module__.split('.')[-1]
        component_name = fn.__name__

        if hasattr(mod, '__all__'):
            mod.__all__.append(component_name)  # type: ignore
        else:
            mod.__all__ = [component_name]  # type: ignore

        if component_name in self._components:
            raise ValueError(
                f'{self.family} {component_name} already registered by {self._components[component_name].module}'
            )

        self._components[component_name] = RegistryEntry(
            name=component_name,
            entrypoint=fn,
            default_config=default_config,
            module=module_name,
        )
        return fn

    def make_decorator(self):
        """
        Returns a register-style decorator bound to this registry,
        supporting both @register_x and @register_x(config=...).
        """
        def register(fn=None, *, config=None):
            def decorator(inner_fn):
                return self.register(inner_fn, default_config=config)
            return decorator(fn) if fn is not None else decorator
        return register

    def list_component(self, module: Optional[str] = None, filter: Optional[str] = None) -> List[str]:
        if module is not None:
            components = [name for name, entry in self._components.items() if entry.module == module]
        else:
            components = list(self._components.keys())

        if filter not in (None, ""):
            try:
                pattern = re.compile(filter, flags=re.IGNORECASE)
            except re.error:
                # Fall back to literal substring semantics when the regex is invalid.
                pattern = re.compile(re.escape(filter), flags=re.IGNORECASE)
            components = [name for name in components if pattern.search(name)]
        return sorted(components)

    def is_registered(self, name: str) -> bool:
        return name in self._components

    def get_entrypoint(self, name: str) -> Callable[..., Any]:
        if name not in self._components:
            raise ValueError(f'{self.family} {name} is not registered')
        return self._components[name].entrypoint

    def get_module(self, name: str) -> str:
        if name not in self._components:
            raise ValueError(f'{self.family} {name} is not registered')
        return self._components[name].module

    def get_default_config(self, name: str) -> Any:
        if name not in self._components:
            raise ValueError(f'{self.family} {name} is not registered')
        if self._components[name].default_config is None:
            raise ValueError(f'{self.family} {name} does not have a default config')
        return self._components[name].default_config
