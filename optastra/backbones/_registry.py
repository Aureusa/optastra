from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("backbone")
register_backbone = _registry.make_decorator()
