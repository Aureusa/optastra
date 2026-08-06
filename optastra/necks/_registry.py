from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("neck")
register_neck = _registry.make_decorator()
