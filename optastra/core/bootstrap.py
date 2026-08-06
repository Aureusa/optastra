def bootstrap() -> None:
    """Import every family's base module so its Factory subclass registers
    itself into core.factory.FACTORIES. Call once at process/import start."""
    from ..backbones import base as _  # noqa: F401
    from ..necks import base as _      # noqa: F401
    from ..heads import base as _      # noqa: F401
    from ..tasks import base as _      # noqa: F401
    from ..algorithms import base as _ # noqa: F401
    from ..proposal_generators import base as _  # noqa: F401
    from ..region_extractors import base as _    # noqa: F401
    from ..architectures import base as _        # noqa: F401
    from ..optim import base as _, scheduler_base as _  # noqa: F401
