import warnings
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field

from ..base import Algorithm, MLPHeadConfig, build_encoder, coerce_mlp_config
from .model import SimCLRModel
from ...backbones.base import Backbone
from ...core.component_ref import ComponentRef, ComponentRefConfigMixin, component_field
from ...necks.base import Neck


__all__ = ["SimCLRTask", "SimCLRConfig", "nt_xent_loss"]


@dataclass
class SimCLRConfig(ComponentRefConfigMixin):
    # --- model (see SimCLRTask.build_model) ---
    backbone: ComponentRef = component_field(Backbone, default_name="resnet50")
    neck: ComponentRef | None = component_field(Neck, optional=True)   # None -> global_avg_pool for CNNs, none for ViTs
    projector: MLPHeadConfig = field(default_factory=lambda: MLPHeadConfig(hidden_dim=2048, out_dim=128))

    # --- loss ---
    temperature: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.projector = coerce_mlp_config(self.projector)


def nt_xent_loss(embeddings: list[torch.Tensor], temperature: float) -> torch.Tensor:
    """
    NT-Xent (normalized temperature-scaled cross entropy, Chen et al. 2020)
    over V >= 2 views of the same N images.

    All V*N embeddings are compared with each other (a (VN, VN) cosine
    similarity matrix / temperature). For every anchor, the positives are
    the other V-1 views of the same image, and the negatives are *every*
    view of every other image -- both from other views and from the
    anchor's own view. The anchor itself is excluded from the softmax.

        loss_i = -1/(V-1) * sum_{p in pos(i)} log( exp(s_ip) / sum_{k != i} exp(s_ik) )

    averaged over all V*N anchors. For V = 2 this is exactly SimCLR's loss
    (symmetric: each view is an anchor once).

    :param embeddings: V tensors of shape (N, D), row n of every tensor is the same image.
    :param temperature: softmax temperature tau.
    """
    num_views, n = len(embeddings), embeddings[0].shape[0]
    # float32 regardless of any autocast around the loss: exp(sim / tau) is
    # too sensitive to bf16 rounding
    z = F.normalize(torch.cat(embeddings, dim=0).float(), dim=-1)   # (V*N, D)
    logits = z @ z.T / temperature                                    # (V*N, V*N)

    self_mask = torch.eye(num_views * n, dtype=torch.bool, device=z.device)
    log_prob = logits.masked_fill(self_mask, float("-inf")).log_softmax(dim=1)

    image_ids = torch.arange(n, device=z.device).repeat(num_views)   # which image each row is
    pos_mask = (image_ids[:, None] == image_ids[None, :]) & ~self_mask
    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / (num_views - 1)
    return -pos_log_prob.mean()


class SimCLRTask(Algorithm):
    """
    SimCLR (Chen et al., 2020): contrastive learning with in-batch negatives,
    no momentum teacher. Works with any number of views >= 2.

        algo = Algorithm.create("simclr", backbone="resnet18", temperature=0.2)
        model = algo.build_model()   # a SimCLRModel

    The model must return one (N, D) projection per view (SimCLRModel does).
    """
    min_views = 2
    collate = "multiview"

    def __init__(self, cfg: SimCLRConfig = SimCLRConfig()):
        self.cfg = cfg
        self.temperature = cfg.temperature

    def build_model(self) -> SimCLRModel:
        backbone, neck, embed_dim = build_encoder(self.cfg.backbone, self.cfg.neck)
        return SimCLRModel(backbone, neck, self.cfg.projector.build(embed_dim))

    def forward_model(self, model, inputs: list) -> list:
        return model(inputs)

    def validate_predictions(self, raw_preds) -> None:
        if not isinstance(raw_preds, (list, tuple)) or len(raw_preds) < self.min_views:
            raise ValueError(
                f"SimCLRTask expects the model to return a list of >= {self.min_views} "
                f"per-view embeddings, got {type(raw_preds).__name__}."
            )

    def compute_losses(self, raw_preds: list, targets) -> dict[str, torch.Tensor]:
        return {"nt_xent_loss": nt_xent_loss(list(raw_preds), self.temperature)}

    def reduce_losses(self, losses):
        return losses["nt_xent_loss"]


@Algorithm.register(config=SimCLRConfig())
def simclr(cfg: SimCLRConfig) -> SimCLRTask:
    return SimCLRTask(cfg)


@Algorithm.register(config=SimCLRConfig())
def simclr_no_momentum(cfg: SimCLRConfig) -> SimCLRTask:
    """Deprecated alias of `simclr`."""
    warnings.warn("'simclr_no_momentum' is deprecated, use 'simclr'.", DeprecationWarning, stacklevel=3)
    return SimCLRTask(cfg)
