import math
import torch, torch.nn.functional as F
from dataclasses import dataclass, field

from ..base import Algorithm, MLPHeadConfig, build_encoder, coerce_mlp_config
from .model import BYOLModel
from ...backbones.base import Backbone
from ...core.component_ref import ComponentRef, ComponentRefConfigMixin, component_field
from ...necks.base import Neck


__all__ = ["BYOLTask", "BYOLConfig"]


@dataclass
class BYOLConfig(ComponentRefConfigMixin):
    # --- model (see BYOLTask.build_model) ---
    backbone: ComponentRef = component_field(Backbone, default_name="resnet50")
    neck: ComponentRef | None = component_field(Neck, optional=True)   # None -> global_avg_pool for CNNs, none for ViTs
    projector: MLPHeadConfig = field(default_factory=lambda: MLPHeadConfig(hidden_dim=4096, out_dim=256))
    predictor: MLPHeadConfig = field(default_factory=lambda: MLPHeadConfig(hidden_dim=4096, out_dim=256))

    # --- target EMA: momentum follows a cosine schedule base -> final ---
    base_momentum: float = 0.996
    final_momentum: float = 1.0
    total_steps: int | None = None   # None -> the Trainer's max_iter

    def __post_init__(self):
        super().__post_init__()
        self.projector = coerce_mlp_config(self.projector)
        self.predictor = coerce_mlp_config(self.predictor)


def _neg_cosine_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # float32 regardless of any autocast around the loss: normalizing in
    # bf16 is too coarse for a loss that lives in [-1, 1]
    p = F.normalize(p.float(), dim=-1)
    z = F.normalize(z.float(), dim=-1)
    return -(p * z).sum(dim=-1)   # per-sample, mean() happens in compute_losses


class BYOLTask(Algorithm):
    """
    BYOL (Grill et al., 2020): the online network predicts the target
    network's projection of the other view; the target is an EMA of the
    online encoder and receives no gradients.

        algo = Algorithm.create("byol", backbone="resnet18")
        model = algo.build_model()
        trainer = Trainer(model, algo, optimizer, hooks=algo.hooks())

    `algo.hooks()` is required: it runs the target EMA update after every
    optimizer step and checkpoints the momentum schedule's step counter.
    """
    min_views = 2
    collate = "multiview"

    def __init__(self, cfg: BYOLConfig = BYOLConfig()):
        self.cfg = cfg
        self.step = 0   # optimizer steps taken so far -- drives the momentum schedule

    def build_model(self) -> BYOLModel:
        backbone, neck, embed_dim = build_encoder(self.cfg.backbone, self.cfg.neck)
        projector = self.cfg.projector.build(embed_dim)
        predictor = self.cfg.predictor.build(self.cfg.projector.out_dim)
        return BYOLModel(backbone, neck, projector, predictor)

    # --- target EMA -----------------------------------------------------
    def momentum_at(self, step: int, total_steps: int | None) -> float:
        """BYOL's cosine schedule, tau = 1 - (1 - tau_base) * (cos(pi * k / K) + 1) / 2,
        generalized to end at `final_momentum` instead of 1. Constant
        `base_momentum` when the total number of steps is unknown."""
        base, final = self.cfg.base_momentum, self.cfg.final_momentum
        if not total_steps:
            return base
        progress = min(step / total_steps, 1.0)
        return final - (final - base) * (math.cos(math.pi * progress) + 1) / 2

    def after_optimizer_step(self, state) -> None:
        model = getattr(state.model, "_orig_mod", state.model)   # unwrap torch.compile
        if not isinstance(model, BYOLModel):
            raise TypeError(f"BYOLTask expects a BYOLModel, got {type(model).__name__}.")
        momentum = self.momentum_at(self.step, self.cfg.total_steps or state.max_iter)
        model.update_target(momentum)
        self.step += 1
        state.storage.put_scalar("byol_momentum", momentum)

    def state_dict(self) -> dict:
        return {"step": self.step}

    def load_state_dict(self, state: dict) -> None:
        self.step = state["step"]

    # --- loss -------------------------------------------------------------

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


@Algorithm.register(config=BYOLConfig())
def byol(cfg: BYOLConfig) -> BYOLTask:
    return BYOLTask(cfg)
