import torch.nn as nn
from ...backbones.base import Backbone
from ...necks.base import Neck
from ..base import encode_pooled


__all__ = ["SimCLRModel"]


class SimCLRModel(nn.Module):
    """Backbone + optional neck + projection head. The projector is
    algorithm-specific -- it doesn't belong in the shared heads/ registry
    because nothing outside SimCLR consumes an "SimCLR projection".

    The backbone (+ neck) must produce `FeatureMaps.pooled`: use a pooling
    neck after a CNN, or no neck after a ViT (SimCLRTask.build_model picks
    this for you)."""

    def __init__(self, backbone: Backbone, neck: Neck | None, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.projector = projector

    def _encode(self, x):
        return encode_pooled(self.backbone, self.neck, x)

    def forward(self, views: list) -> list:
        return [self.projector(self._encode(v)) for v in views]
