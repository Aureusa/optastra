import torch, torch.nn.functional as F
from dataclasses import dataclass
from ..base import Algorithm
from .._registry import register_algorithm


__all__ = ["SimCLRTask"]


@dataclass
class SimCLRConfig:
    temperature: float = 0.5


class SimCLRTask(Algorithm):
    """
    Simple SimCLR implementation, for demonstration purposes. No momentum teacher. 
    The model is expected to be a SimCLRModel"""
    min_views = 2

    def __init__(self, cfg: SimCLRConfig = SimCLRConfig()):
        self.cfg = cfg
        self.temperature = cfg.temperature

    def forward_model(self, model, inputs: list) -> list:
        return model(inputs)

    def validate_predictions(self, raw_preds) -> None:
        """No-op for SimCLR, since the model output is just a list of embeddings."""
        pass

    def compute_losses(self, raw_preds: list, targets) -> dict[str, torch.Tensor]:
        z1, z2 = (F.normalize(z, dim=-1) for z in raw_preds)   # NT-Xent, two-view case
        logits = z1 @ z2.T / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return {"nt_xent_loss": loss}

    def reduce_losses(self, losses):
        return losses["nt_xent_loss"]


@register_algorithm(config=SimCLRConfig())
def simclr_no_momentum(cfg: SimCLRConfig) -> SimCLRTask:
    return SimCLRTask(cfg)
