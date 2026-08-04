from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("architecture")
register_architecture = _registry.make_decorator()
