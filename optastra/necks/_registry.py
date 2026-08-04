from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("neck")
register_neck = _registry.make_decorator()
