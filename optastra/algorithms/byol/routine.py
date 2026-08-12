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
        return {"byol_loss": 0.5 * (loss_1to2 + loss_2to1).mean()}

    def reduce_losses(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return losses["byol_loss"]

    def compute_metrics(self, raw_preds, targets) -> dict[str, float]:
        """
        Compute metrics for monitoring, e.g., the standard deviation of the online and target representations.
        """
        # Normalize representations
        online_z1 = F.normalize(raw_preds["online_z1"], dim=-1)
        online_z2 = F.normalize(raw_preds["online_z2"], dim=-1)

        target_z1 = F.normalize(raw_preds["target_z1"], dim=-1)
        target_z2 = F.normalize(raw_preds["target_z2"], dim=-1)

        # ------------------------------------------------------------
        # 1. Positive-pair similarity
        #
        # Same image, two different augmentations.
        # This should generally increase during training.
        # Value in the range [-1, 1], with 1 being perfect alignment.
        # ------------------------------------------------------------
        pos_cos = 0.5 * (
            (online_z1 * target_z2).sum(dim=-1).mean()
            + (online_z2 * target_z1).sum(dim=-1).mean()
        )

        # ------------------------------------------------------------
        # 2. Negative / different-image similarity
        #
        # Compare online representation of image i with target
        # representation of image j, j != i.
        #
        # We use a cyclic shift so that no image is compared with itself.
        # This should NOT converge toward 1.
        # ------------------------------------------------------------
        neg_target_z1 = torch.roll(target_z1, shifts=1, dims=0)
        neg_target_z2 = torch.roll(target_z2, shifts=1, dims=0)

        neg_cos = 0.5 * (
            (online_z1 * neg_target_z1).sum(dim=-1).mean()
            + (online_z2 * neg_target_z2).sum(dim=-1).mean()
        )

        # ------------------------------------------------------------
        # 3. Representation standard deviation
        #
        # Average std across embedding dimensions.
        # Collapse -> approximately 0.
        # ------------------------------------------------------------
        online_std = 0.5 * (
            online_z1.std(dim=0).mean()
            + online_z2.std(dim=0).mean()
        )

        target_std = 0.5 * (
            target_z1.std(dim=0).mean()
            + target_z2.std(dim=0).mean()
        )

        # ------------------------------------------------------------
        # 4. Effective rank
        #
        # Measures how many dimensions of the embedding are actually
        # being used.
        #
        # 1 = complete collapse
        # D = approximately full-rank representation.
        # ------------------------------------------------------------
        z = torch.cat([online_z1, online_z2], dim=0).float()
        z = z - z.mean(dim=0, keepdim=True)

        cov = z.T @ z / max(z.shape[0] - 1, 1)

        eigenvalues = torch.linalg.eigvalsh(cov).clamp_min(1e-12)

        p = eigenvalues / eigenvalues.sum()

        effective_rank = torch.exp(
            -(p * torch.log(p)).sum()
        )

        return {
            "byol_pos_cos": pos_cos.item(),
            "byol_neg_cos": neg_cos.item(),
            "online_std": online_std.item(),
            "target_std": target_std.item(),
            "online_eff_rank": effective_rank.item(),
        }


@register_task(config=BYOLConfig())
def byol(cfg: BYOLConfig) -> BYOLTask:
    return BYOLTask(cfg)
