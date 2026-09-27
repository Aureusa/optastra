import copy
import torch, torch.nn as nn
from ...backbones.base import Backbone
from ...necks.base import Neck
from ..base import encode_pooled


__all__ = ["BYOLModel"]


class BYOLModel(nn.Module):
    """Online: backbone+neck+projector+predictor, trained by backprop.
    Target: backbone+neck+projector (no predictor), EMA-updated only --
    built as a deep copy of the online encoder so architectures always
    match exactly, then detached from autograd.

    The target is moved towards the online network by `update_target(m)`,
    which BYOLTask calls after every optimizer step (see Algorithm.hooks())."""

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

    def _online_target_pairs(self):
        pairs = [(self.online_backbone, self.target_backbone), (self.online_projector, self.target_projector)]
        if self.online_neck is not None:
            pairs.append((self.online_neck, self.target_neck))
        return pairs

    def _target_parameters(self):
        for _, target in self._online_target_pairs():
            yield from target.parameters()

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """EMA step: target = momentum * target + (1 - momentum) * online,
        for every parameter AND every floating-point buffer (e.g. BatchNorm
        running mean/var). Integer buffers (BN's num_batches_tracked) are
        copied, since averaging a counter is meaningless."""
        for online, target in self._online_target_pairs():
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.lerp_(p_o, 1.0 - momentum)
            for b_o, b_t in zip(online.buffers(), target.buffers()):
                if b_t.is_floating_point():
                    b_t.lerp_(b_o, 1.0 - momentum)
                else:
                    b_t.copy_(b_o)

    def forward(self, views: list[torch.Tensor]) -> dict:
        v1, v2 = views   # BYOL is defined for exactly 2 views
        online_z1 = self.predictor(self.online_projector(encode_pooled(self.online_backbone, self.online_neck, v1)))
        online_z2 = self.predictor(self.online_projector(encode_pooled(self.online_backbone, self.online_neck, v2)))
        with torch.no_grad():
            target_z1 = self.target_projector(encode_pooled(self.target_backbone, self.target_neck, v1))
            target_z2 = self.target_projector(encode_pooled(self.target_backbone, self.target_neck, v2))
        return {
            "online_z1": online_z1, "online_z2": online_z2,
            "target_z1": target_z1.detach(), "target_z2": target_z2.detach(),
        }
