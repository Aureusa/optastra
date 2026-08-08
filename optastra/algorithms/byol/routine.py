# algorithms/byol/task.py
import torch, torch.nn.functional as F
from dataclasses import dataclass
from ..base import Algorithm
from ...tasks._registry import register_task


@dataclass
class BYOLConfig:
    momentum: float = 0.996   # EMA rate for target network -- typically annealed to 1.0 over training

def _neg_cosine_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1)   # per-sample, mean() happens in compute_losses


class BYOLTask(Algorithm):
    min_views = 2
    collate = "multiview"
    def __init__(self, cfg: BYOLConfig = BYOLConfig()):
        self.cfg = cfg

    def forward_model(self, model, inputs: list[torch.Tensor]) -> dict:
        return model(inputs)   # inputs is the list of 2 views; model returns the dict from BYOLModel.forward

    def validate_predictions(self, raw_preds: dict) -> None:
        required = {"online_z1", "online_z2", "target_z1", "target_z2"}
        missing = required - raw_preds.keys()
        if missing:
            raise ValueError(f"BYOLTask requires {missing} from the model output.")

    def compute_losses(self, raw_preds: dict, targets) -> dict[str, torch.Tensor]:
        # symmetric loss: predict view2's target from view1's online, and vice versa
        loss_1to2 = _neg_cosine_sim(raw_preds["online_z1"], raw_preds["target_z2"])
        loss_2to1 = _neg_cosine_sim(raw_preds["online_z2"], raw_preds["target_z1"])
        return {"byol_loss": (loss_1to2 + loss_2to1).mean()}

    def reduce_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return losses["byol_loss"]


@register_task(config=BYOLConfig())
def byol(cfg: BYOLConfig) -> BYOLTask:
    return BYOLTask(cfg)
