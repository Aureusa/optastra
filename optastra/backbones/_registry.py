from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("backbone")
register_backbone = _registry.make_decorator()
