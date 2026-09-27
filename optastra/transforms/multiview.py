from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import Transform
from .compose import Compose
from ..core.component_ref import coerce_to_ref
from ..data.sample import Sample


__all__ = ["MultiViewTransform"]


class MultiViewTransform(Transform):
    """
    Turns one Sample into several independently augmented views of it, for
    self-supervised algorithms: returns a Sample whose `views` holds one
    image per view pipeline (and `image=None`), which the "multiview"
    collate stacks into `batch["views"]`.

        # one pipeline per view (e.g. BYOL's asymmetric augmentations, or
        # DINO-style multi-crop: 2 global + N small local crops)
        MultiViewTransform([view1_pipeline, view2_pipeline])

        # the same random pipeline applied n times (SimCLR)
        MultiViewTransform(pipeline, n_views=2)

    Each pipeline runs on its own copy of the input sample, so in-place
    transforms can't leak between views. `target` and `meta` are kept from
    the input sample.
    """

    def __init__(self, transforms: Transform | Sequence[Transform], n_views: int | None = None):
        transforms = [transforms] if isinstance(transforms, Transform) else list(transforms)
        if n_views is not None:
            if len(transforms) != 1:
                raise ValueError(f"n_views={n_views} repeats a single pipeline, got {len(transforms)}.")
            transforms = transforms * n_views
        if not transforms:
            raise ValueError("MultiViewTransform needs at least one view pipeline.")
        self.transforms = transforms

    def __call__(self, sample: Sample) -> Sample:
        views = []
        for transform in self.transforms:
            view = Sample(
                image=sample.image.clone(),
                target=copy.deepcopy(sample.target),
                meta=copy.deepcopy(sample.meta),
            )
            views.append(transform(view).image)
        return Sample(image=None, views=views, target=sample.target, meta=sample.meta)


@dataclass
class MultiViewConfig:
    # One entry per view. Each entry is a transform ref ("name",
    # ("name", {overrides}), {"name": ..., "overrides": {...}}) or a list of
    # refs applied in order. Inside YAML, use the dict form for overrides.
    views: list[Any] = field(default_factory=list)
    # Only used when `views` has 0 or 1 entries: repeat that one pipeline
    # this many times (with no `views`, each view is an unmodified copy).
    n_views: int | None = 2


def _build_pipeline(entry: Any) -> Transform:
    if isinstance(entry, list):
        return Compose([coerce_to_ref(ref).resolve(Transform) for ref in entry])
    return coerce_to_ref(entry).resolve(Transform)


@Transform.register(config=MultiViewConfig())
def multi_view(cfg: MultiViewConfig) -> MultiViewTransform:
    pipelines = [_build_pipeline(entry) for entry in cfg.views] or [Compose([])]
    n_views = cfg.n_views if len(pipelines) == 1 else None
    return MultiViewTransform(pipelines, n_views=n_views)
