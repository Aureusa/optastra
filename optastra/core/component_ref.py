from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any
from dataclasses import is_dataclass, fields

from .factory import Factory


@dataclass
class ComponentRef:
    """
    Names a registered component + overrides -- never a resolved config
    object. This is what every .create(name, **overrides) call actually
    consumes, so it's always both serializable and reconstructible."""
    name: str = field(default_factory=str)
    overrides: dict[str, Any] = field(default_factory=dict)
    factory: Factory | None = field(default=None, repr=False, compare=False)

    def resolve(self, factory: Factory, **extras) -> Any:
        """Resolve the component reference to a concrete object."""
        if self.factory is not None:
            return self.factory.create(self.name, **self.overrides, **extras)
        else:
            return factory.create(self.name, **self.overrides, **extras)

    def resolve_config(
        self,
        fallback_factory: Factory | None = None,
    ) -> dict[str, Any]:
        """
        Convert this reference into a fully expanded serializable config.
        Output contains only effective merged fields.
        """
        factory = self.factory or fallback_factory
        if factory is None:
            overrides = _serialize_config(self.overrides)
            return {
                "name": self.name,
                **overrides,
            }

        default = factory.get_default_config(self.name)

        default_serialized = _serialize_config(default) if default is not None else {}
        overrides_serialized = _serialize_config(self.overrides)
        resolved = _deep_merge(default_serialized, overrides_serialized)

        return {
            "name": self.name,
            **resolved,
        }


def _deep_merge(base: Any, patch: Any) -> Any:
    if isinstance(base, dict) and isinstance(patch, dict):
        merged = {k: _serialize_config(v) for k, v in base.items()}
        for k, v in patch.items():
            if k in merged:
                merged[k] = _deep_merge(merged[k], v)
            else:
                merged[k] = _serialize_config(v)
        return merged
    return _serialize_config(patch)


def _serialize_config(value: Any) -> Any:
    """
    Convert dataclasses and ComponentRefs recursively
    into YAML-safe Python objects.
    """

    if isinstance(value, ComponentRef):
        return value.resolve_config()

    if is_dataclass(value):
        return {
            f.name: _serialize_field(f.name, getattr(value, f.name))
            for f in fields(value)
            if f.name != "factory"
        }

    if isinstance(value, dict):
        return {
            k: _serialize_field(str(k), v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _serialize_config(v)
            for v in value
        ]

    if isinstance(value, type):
        return value.__name__

    return value


def _serialize_field(field_name: str, value: Any) -> Any:
    if isinstance(value, ComponentRef) and value.factory is None:
        inferred_factory = _infer_factory_for_field(field_name)
        if inferred_factory is not None:
            value = replace(value, factory=inferred_factory)
    return _serialize_config(value)


def _infer_factory_for_field(field_name: str) -> Factory | None:
    from ..architectures.base import Architecture
    from ..backbones.base import Backbone
    from ..heads.base import Head
    from ..necks.base import Neck
    from ..optim.base import Optimizer
    from ..optim.scheduler_base import Scheduler
    from ..proposal_generators.base import ProposalGenerator
    from ..region_extractors.base import RegionExtractor
    from ..tasks.base import Task
    from ..detection.base_criterion import DetectionCriterion
    from ..detection.base_matcher import Matcher
    from ..detection.base_postprocessor import Postprocessor
    from ..detection.base_sampler import Sampler

    by_field_name: dict[str, Factory] = {
        "architecture": Architecture,
        "task": Task,
        "optimizer": Optimizer,
        "scheduler": Scheduler,
        "backbone": Backbone,
        "neck": Neck,
        "proposal_generator": ProposalGenerator,
        "region_extractor": RegionExtractor,
        "mask_region_extractor": RegionExtractor,
        "roi_box_head": Head,
        "mask_head": Head,
        "criterion": DetectionCriterion,
        "postprocessor": Postprocessor,
        "matcher": Matcher,
        "sampler": Sampler,
    }
    return by_field_name.get(field_name)