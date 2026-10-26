from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..core.factory import SpecFactory
from ..core.registry import FamilyRegistry
from ..nn.features import FeatureMaps, FeatureSpec


__all__ = ["Decoder", "QuerySet"]


@dataclass
class QuerySet:
    """Queries supplied from outside the decoder, e.g. priors at known positions.

    content           (B, K, D) initial query content, or None (zeros, like DETR's)
    reference_points  (B, K, 2) (x, y) as fractions of each image's size, or None
    mask              (B, K) bool, real queries (pad ragged sets with False), or None (all real)
    """
    content: torch.Tensor | None = None
    reference_points: torch.Tensor | None = None
    mask: torch.Tensor | None = None

    def __len__(self) -> int:
        for t in (self.content, self.reference_points, self.mask):
            if t is not None:
                return int(t.shape[1])
        return 0


class Decoder(nn.Module, SpecFactory["Decoder"], ABC):
    """A decoder turns features into a *set of tokens*.

    Where a Neck maps FeatureMaps to FeatureMaps of the same kind (spatial maps
    in, spatial maps out), a decoder reads the features as a memory and emits a
    sequence of tokens that are not tied to pixel locations -- object queries
    (DETR), mask queries, caption tokens. The tokens come back as
    ``FeatureMaps.patch_tokens`` (B, M, D) with ``out_spec.embed_dim = D``, so any
    Head that reads tokens can sit on top; ``FeatureMaps.extra`` carries what
    the decoder knows about them (``optastra.decoders.keys``).
    """

    _registry = FamilyRegistry("decoder")
    out_spec: FeatureSpec

    @classmethod
    def _post_create(cls, decoder: "Decoder") -> "Decoder":
        if not isinstance(getattr(decoder, "out_spec", None), FeatureSpec) or decoder.out_spec.embed_dim is None:
            raise ValueError(f"{type(decoder).__name__} must set out_spec = FeatureSpec(embed_dim=...).")
        return decoder

    def forward(self, features: FeatureMaps, queries: QuerySet | None = None) -> FeatureMaps:
        raise NotImplementedError
