from optastra.core.registry import ComponentRegistry


_registry = ComponentRegistry("proposal_generator")
register_proposal_generator = _registry.make_decorator()
