from optastra.core.registry import FamilyRegistry


_registry = FamilyRegistry("proposal_generator")
register_proposal_generator = _registry.make_decorator()
