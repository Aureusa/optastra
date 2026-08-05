from __future__ import annotations
from dataclasses import dataclass

from .base import Neck
from ._registry import register_neck
from ..nn.features import FeatureMaps, FeatureSpec
from ..nn.blocks.readout.pooling import (
    GlobalAvgPool2d,
    GlobalMaxPool2d,
    GeneralizedMeanPooling,
    TokenPooling,
)


__all__ = ["GlobalPool", "GeM", "TokenPool"]


def _check_stage_in_spec(stage: str, in_spec: FeatureSpec) -> None:
    """Check if the specified stage exists in the input FeatureSpec."""
    if stage not in in_spec.channels:
        raise ValueError(
            f"Stage '{stage}' not found in in_spec.channels: {list(in_spec.channels.keys())}"
        )


##############################################################
#################### Global Mean Pooling #####################
##############################################################


@dataclass
class GlobalPoolConfig:
    pool_type: str = "avg"  # "avg" or "max"
    stage: str | None = None  # None = use the deepest/highest-stride stage


class GlobalPool(Neck):
    """Adapts spatial feature maps (e.g. from a CNN backbone) into a single
    pooled embedding, so any downstream head can require embed_dim without
    caring whether it came from a CNN's spatial maps or a transformer's CLS token."""

    def __init__(self, in_spec: FeatureSpec, cfg: GlobalPoolConfig):
        super().__init__()
        self.cfg = cfg
        in_spec.require("channels", "strides") # Ensure that the in_spec has both channels and strides defined

        self.stage = cfg.stage or max(in_spec.strides, key=in_spec.strides.get)
        if cfg.pool_type == "avg":
            self.pool = GlobalAvgPool2d()
        elif cfg.pool_type == "max":
            self.pool = GlobalMaxPool2d()
        else:
            raise ValueError(f"{self.__class__.__name__} does not support pool_type: {cfg.pool_type}")

        _check_stage_in_spec(self.stage, in_spec)
        
        self.out_spec = FeatureSpec(embed_dim=in_spec.channels[self.stage])

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        x = self.pool(features.feature_maps[self.stage]).flatten(1)
        return FeatureMaps(pooled=x)


##############################################################
################# Generalized Mean Pooling ###################
##############################################################


@dataclass
class GeMConfig:
    stage: str | None = None  # None = use the deepest/highest-stride stage
    p: float = 3.0  # GeM pooling parameter
    eps: float = 1e-6  # Small value to avoid division by zero


class GeM(Neck):
    """Generalized Mean Pooling (GeM) layer."""

    def __init__(self, in_spec: FeatureSpec, cfg: GeMConfig):
        super().__init__()
        self.cfg = cfg
        in_spec.require("channels", "strides") # Ensure that the in_spec has both channels and strides defined

        self.stage = cfg.stage or max(in_spec.strides, key=in_spec.strides.get)

        _check_stage_in_spec(self.stage, in_spec)
        
        self.pool = GeneralizedMeanPooling(p=cfg.p, eps=cfg.eps)
        
        self.out_spec = FeatureSpec(embed_dim=in_spec.channels[self.stage])

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        x = self.pool(features.feature_maps[self.stage]).flatten(1)
        return FeatureMaps(pooled=x)


##############################################################
###################### Token Pooling #########################
##############################################################

# TODO: Migh need to move it to a seperate file, like transformer_pool.py
@dataclass
class TokenPoolConfig: # Experimental still, not fully integrated into the framework yet
    stage: str | None = None  # None = use the deepest/highest-stride stage


class TokenPool(Neck):
    """Token Pooling layer for pooling token embeddings."""

    def __init__(self, in_spec: FeatureSpec, cfg: TokenPoolConfig):
        super().__init__()
        self.cfg = cfg
        in_spec.require("channels", "strides") # Ensure that the in_spec has both channels and strides defined

        self.stage = cfg.stage or max(in_spec.strides, key=in_spec.strides.get)

        _check_stage_in_spec(self.stage, in_spec)

        self.pool = TokenPooling()
        
        self.out_spec = FeatureSpec(embed_dim=in_spec.channels[self.stage])

    def forward(self, features: FeatureMaps) -> FeatureMaps:
        x = self.pool(features.feature_maps[self.stage]).flatten(1)
        return FeatureMaps(pooled=x)


pool_configs = {
    "global_avg_pool": GlobalPoolConfig(pool_type="avg"),
    "global_max_pool": GlobalPoolConfig(pool_type="max"),
    "gem_pool": GeMConfig(p=3.0, eps=1e-6),
    "token_pool": TokenPoolConfig(stage=None),
}


@register_neck(config=pool_configs["global_avg_pool"])
def global_avg_pool(in_spec: FeatureSpec, cfg: GlobalPoolConfig) -> GlobalPool:
    """Factory function to create a Global Average Pooling neck."""
    return GlobalPool(in_spec, cfg)


@register_neck(config=pool_configs["global_max_pool"])
def global_max_pool(in_spec: FeatureSpec, cfg: GlobalPoolConfig) -> GlobalPool:
    """Factory function to create a Global Max Pooling neck."""
    return GlobalPool(in_spec, cfg)


@register_neck(config=pool_configs["gem_pool"])
def gem_pool(in_spec: FeatureSpec, cfg: GeMConfig) -> GeM:
    """Factory function to create a Generalized Mean Pooling neck."""
    return GeM(in_spec, cfg)


@register_neck(config=pool_configs["token_pool"])
def token_pool(in_spec: FeatureSpec, cfg: TokenPoolConfig) -> TokenPool:
    """Factory function to create a Token Pooling neck."""
    return TokenPool(in_spec, cfg)
