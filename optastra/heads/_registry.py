from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("head")
register_head = _registry.make_decorator()
