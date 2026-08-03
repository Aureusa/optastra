import torch.nn as nn
from ...backbones.base import Backbone
from ...necks.base import Neck


__all__ = ["SimCLRModel"]


class SimCLRModel(nn.Module):
    """Backbone + optional neck + projection head. The projector is
    algorithm-specific -- it doesn't belong in the shared heads/ registry
    because nothing outside SimCLR consumes an "SimCLR projection"."""

    def __init__(self, backbone: Backbone, neck: Neck, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.projector = projector

    def _encode(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats.pooled  # assumes a pooling neck already in the chain

    def forward(self, views: list) -> list:
        return [self.projector(self._encode(v)) for v in views]
    