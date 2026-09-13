"""Optastra: a modular computer vision framework built on PyTorch.

Importing this package registers every built-in component family
(backbones, necks, heads, tasks, algorithms, architectures, optimizers,
schedulers, proposal generators, region extractors, transforms) so that
``Backbone.create("resnet50")`` and friends work immediately -- no manual
``bootstrap()`` call required.

See ``docs/getting-started.md`` for a walkthrough and ``docs/concepts.md``
for how the Registry -> Factory -> ComponentRef pieces fit together.
"""
from __future__ import annotations

from .core.bootstrap import bootstrap

bootstrap()

from .core.factory import (
    Factory,
    SpecFactory,
    list_all_registered_components,
    list_all_registered_families,
    get_component_parameters,
)
from .core.component_ref import ComponentRef, component_field
from .core.experiment import ExperimentConfig
from .core.build import (
    build_experiment_from_config,
    build_sequential_model,
)
from .core.describe import resolve_experiment, print_experiment, count_parameters

from .backbones import Backbone
from .necks import Neck
from .heads import Head
from .tasks import Task
from .algorithms.base import Algorithm
from .architectures import Architecture
from .proposal_generators import ProposalGenerator
from .region_extractors import RegionExtractor
from .optim import Optimizer, Scheduler
from .transforms import Transform, BatchTransform
from .training import Trainer
from .data import Sample, build_dataloader
from .data.collate import CollateFn

__all__ = [
    "bootstrap",
    "Factory",
    "SpecFactory",
    "list_all_registered_components",
    "list_all_registered_families",
    "get_component_parameters",
    "ComponentRef",
    "component_field",
    "ExperimentConfig",
    "build_experiment_from_config",
    "build_sequential_model",
    "resolve_experiment",
    "print_experiment",
    "count_parameters",
    "Backbone",
    "Neck",
    "Head",
    "Task",
    "Algorithm",
    "Architecture",
    "ProposalGenerator",
    "RegionExtractor",
    "Optimizer",
    "Scheduler",
    "Transform",
    "BatchTransform",
    "Trainer",
    "Sample",
    "build_dataloader",
    "CollateFn",
]
