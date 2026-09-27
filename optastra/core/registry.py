from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, TypeVar


T = TypeVar("T", bound=Callable[..., Any])


@dataclass
class RegistryEntry:
    name: str
    entrypoint: Callable
    default_config: Any
    module: str


class FamilyRegistry:
    """
    Name -> (entrypoint, default config) table for one component family.

    Registration never touches the defining module's namespace (in particular
    its `__all__`): what a module exports is decided by the module itself.
    """

    def __init__(
        self,
        family: str,
    ):
        self.family = family
        self._components: Dict[str, RegistryEntry] = {}

    def register(self, fn: T, *, default_config: Optional[Any] = None, name: Optional[str] = None) -> T:
        """
        Register `fn` under `name` (default: `fn.__name__`). Registering the same
        callable under several names is fine -- that is how variant tables are
        registered in a loop. Returns `fn` unchanged so this works as a decorator.
        """
        component_name = name or fn.__name__
        if component_name in self._components:
            raise ValueError(
                f'{self.family} {component_name} already registered by {self._components[component_name].module}'
            )

        self._components[component_name] = RegistryEntry(
            name=component_name,
            entrypoint=fn,
            default_config=default_config,
            module=fn.__module__.split('.')[-1],
        )
        return fn

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

    def _entry(self, name: str) -> RegistryEntry:
        if name not in self._components:
            raise ValueError(f'{self.family} {name} is not registered')
        return self._components[name]

    def get_entrypoint(self, name: str) -> Callable[..., Any]:
        return self._entry(name).entrypoint

    def get_module(self, name: str) -> str:
        return self._entry(name).module

    def get_default_config(self, name: str) -> Any:
        """The registered default config object, or None for a configless component.
        This is the shared original -- Factory hands out copies."""
        return self._entry(name).default_config
