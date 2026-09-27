from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from ..backbones.base import Backbone
from ..core.component_ref import ComponentRef
from ..core.registry import FamilyRegistry
from ..necks.base import Neck
from ..nn.blocks.readout.mlp import MLP
from ..tasks.base import Task, Stage
from ..training.hooks.base import Hook


__all__ = ["Algorithm", "AlgorithmHook", "MLPHeadConfig", "build_encoder", "encode_pooled"]


class Algorithm(Task):
    """
    A self-supervised pretraining objective.

    Reuses Task's step machinery (same run_step / TaskStepOutput / Trainer
    integration) -- the difference is entirely in what the batch contains
    and what the loss compares: multiple augmented views of the same
    image, no external labels. It is its *own* registry family
    ("algorithm"), so `Algorithm.create(...)` only sees SSL methods and
    `Task.list_all()` never lists them.

    On top of a Task, an Algorithm:
      - builds the model it trains (`build_model()`), from ComponentRefs in
        its config, so swapping the backbone is a config change;
      - may carry per-step state (EMA teacher, momentum/temperature
        schedules), advanced by `after_optimizer_step(state)` and saved /
        restored through `state_dict()` / `load_state_dict()`. Pass
        `algorithm.hooks()` to the Trainer so both happen automatically.
    """
    _registry = FamilyRegistry("algorithm")
    min_views: int = 2   # default minimum number of views required for a batch
    collate = "multiview"

    # --- model construction ---------------------------------------------
    def build_model(self) -> nn.Module:
        """Build the model this algorithm trains, from `self.cfg`."""
        raise NotImplementedError(f"{type(self).__name__} does not implement build_model().")

    # --- per-step state -------------------------------------------------
    def after_optimizer_step(self, state) -> None:
        """Called once after every optimizer step (via AlgorithmHook).
        Override for EMA teachers, schedules, etc. Default: nothing."""

    def hooks(self) -> list[Hook]:
        """Hooks the Trainer needs for this algorithm to train correctly."""
        return [AlgorithmHook(self)]

    def state_dict(self) -> dict:
        """Resumable per-step state (NOT model weights -- those live in the model)."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        return

    # --- Task plumbing --------------------------------------------------
    def validate_batch(self, batch: Mapping[str, Any], stage: Stage = "train"):
        if "views" not in batch:
            raise ValueError("Algorithm batches must contain a 'views' key: list[Tensor].")
        if len(batch["views"]) < self.min_views:
            raise ValueError(f"{type(self).__name__} requires >= {self.min_views} views.")

    def split_inputs_targets(self, batch: Mapping[str, Any], stage: Stage = "train"):
        # Reuse the same views as pseudo-targets so Task.run_step executes
        # the normal train/val loss path without requiring external labels.
        return batch["views"], batch["views"]

    def preprocess_targets(self, raw_targets):
        """Pass-through: algorithms treat views as self-supervised targets."""
        return raw_targets

    def decode_predictions(self, raw_preds: Any) -> Any:
        """No-op for algorithms, since there are no external targets."""
        return raw_preds

    def compute_metrics(self, raw_preds, targets) -> dict[str, float]:
        """No-op for algorithms, since there are no external targets."""
        return {}


class AlgorithmHook(Hook):
    """Calls `algorithm.after_optimizer_step(state)` after every training
    step (the Trainer runs `after_step` right after `optimizer.step()`), and
    stores the algorithm's per-step state in checkpoints -- the Checkpointer
    saves/restores every hook's `state_dict()`.

    Get it from `algorithm.hooks()` rather than building it by hand."""

    # Before anything that reads the updated model or saves it (EvalHook,
    # CheckpointHook at 100), after ResumeHook (0).
    priority = 20

    def __init__(self, algorithm: Algorithm):
        self.algorithm = algorithm

    def after_step(self, state) -> None:
        self.algorithm.after_optimizer_step(state)

    def state_dict(self) -> dict:
        return self.algorithm.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.algorithm.load_state_dict(state)


@dataclass
class MLPHeadConfig:
    """Projector / predictor MLP used by SSL algorithms:
    Linear -> BatchNorm1d -> ReLU -> ... -> Linear."""
    hidden_dim: int = 4096
    out_dim: int = 256
    num_layers: int = 2
    norm: str | None = "batchnorm1d"
    activation: str = "relu"

    def build(self, in_dim: int) -> MLP:
        return MLP(
            in_features=in_dim, hidden_features=self.hidden_dim, out_features=self.out_dim,
            num_layers=self.num_layers, activation=self.activation, norm=self.norm,
        )


def coerce_mlp_config(value: MLPHeadConfig | Mapping[str, Any]) -> MLPHeadConfig:
    """Lets configs (and YAML) pass a plain dict for a projector/predictor."""
    return value if isinstance(value, MLPHeadConfig) else MLPHeadConfig(**value)


def build_encoder(backbone: ComponentRef, neck: ComponentRef | None) -> tuple[Backbone, Neck | None, int]:
    """
    Build the backbone (+ neck) an SSL algorithm trains, and return the size
    of the pooled embedding they produce.

    SSL models read `FeatureMaps.pooled`, so the encoder has to output one:
      - backbones that already pool (ViT: `out_spec.embed_dim` is set, the
        CLS token is returned as `pooled`) need no neck;
      - spatial backbones (CNNs) default to a `global_avg_pool` neck when
        `neck` is None.
    Pass a neck ref explicitly to override either default.
    """
    bb = backbone.resolve(Backbone)
    if neck is None and bb.out_spec.embed_dim is None:
        neck = ComponentRef("global_avg_pool")
    nk = neck.resolve(Neck, in_spec=bb.out_spec) if neck is not None else None
    out_spec = nk.out_spec if nk is not None else bb.out_spec
    out_spec.require("embed_dim")
    return bb, nk, out_spec.embed_dim


def encode_pooled(backbone: nn.Module, neck: nn.Module | None, x: torch.Tensor) -> torch.Tensor:
    """backbone (+ neck) -> the pooled (B, D) embedding SSL projectors consume."""
    feats = backbone(x)
    if neck is not None:
        feats = neck(feats)
    if feats.pooled is None:
        raise ValueError(
            f"{type(neck or backbone).__name__} returned no `pooled` embedding. SSL models need a "
            "pooled output: add a pooling neck (e.g. 'global_avg_pool') after a spatial backbone."
        )
    return feats.pooled
