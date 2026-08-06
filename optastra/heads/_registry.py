from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("head")
register_head = _registry.make_decorator()
