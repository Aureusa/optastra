import copy
import torch, torch.nn as nn
from ...backbones.base import Backbone
from ...necks.base import Neck


class BYOLModel(nn.Module):
    """Online: backbone+neck+projector+predictor, trained by backprop.
    Target: backbone+neck+projector (no predictor), EMA-updated only --
    built as a deep copy of the online encoder so architectures always
    match exactly, then detached from autograd."""

    def __init__(self, backbone: Backbone, neck: Neck | None, projector: nn.Module, predictor: nn.Module):
        super().__init__()
        self.online_backbone = backbone
        self.online_neck = neck
        self.online_projector = projector
        self.predictor = predictor   # online-only, target has no predictor

        self.target_backbone = copy.deepcopy(backbone)
        self.target_neck = copy.deepcopy(neck) if neck is not None else None
        self.target_projector = copy.deepcopy(projector)
        for p in self._target_parameters():
            p.requires_grad_(False)

    def _target_parameters(self):
        mods = [self.target_backbone, self.target_projector]
        if self.target_neck is not None:
            mods.append(self.target_neck)
        for m in mods:
            yield from m.parameters()

    def _encode(self, backbone, neck, projector, x):
        feats = backbone(x)
        if neck is not None:
            feats = neck(feats)
        return projector(feats.pooled)

    def forward(self, views: list[torch.Tensor]) -> dict:
        v1, v2 = views   # BYOL is defined for exactly 2 views
        online_z1 = self.predictor(self._encode(self.online_backbone, self.online_neck, self.online_projector, v1))
        online_z2 = self.predictor(self._encode(self.online_backbone, self.online_neck, self.online_projector, v2))
        with torch.no_grad():
            target_z1 = self._encode(self.target_backbone, self.target_neck, self.target_projector, v1)
            target_z2 = self._encode(self.target_backbone, self.target_neck, self.target_projector, v2)
        return {
            "online_z1": online_z1, "online_z2": online_z2,
            "target_z1": target_z1.detach(), "target_z2": target_z2.detach(),
        }
    