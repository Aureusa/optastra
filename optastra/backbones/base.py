from __future__ import annotations

from abc import ABC
from dataclasses import fields, replace
from typing import Any
import torch
import torch.nn as nn

from ._registry import get_backbone_entrypoint, get_backbone_default_config, list_backbones
from ..nn.features import FeatureMaps, FeatureSpec


__all__ = ["Backbone"]


class Backbone(nn.Module, ABC):
    """A backbone only produces features -- it knows nothing about tasks."""

    out_spec: FeatureSpec  # type: ignore

    @staticmethod
    def _validate_out_spec(backbone: "Backbone") -> None:
        """Ensure every created backbone exposes a valid FeatureSpec."""
        if not isinstance(backbone.out_spec, FeatureSpec):
            raise ValueError(
                f"{backbone.__class__.__name__} must define an 'out_spec' attribute of type FeatureSpec. Check docs for details."
            )

    @classmethod
    def create(cls, name: str, **overrides) -> Backbone:  # Factory method to create a backbone by name
        """Create a backbone by name, optionally loading pretrained weights.

        :param name: Name of the backbone to create.
        :param overwrites: Optional keyword arguments to overwrite the default configuration.
        :return: An instance of the backbone.
        """
        entrypoint = get_backbone_entrypoint(name)
        default_cfg = get_backbone_default_config(name)
        cfg = replace(default_cfg, **overrides)  # raises on unknown fields
        backbone = entrypoint(cfg)
        cls._validate_out_spec(backbone)
        return backbone

    @classmethod
    def describe(cls, name: str) -> dict[str, int]: # Factory method to describe a backbone by name
        """Describe a backbone by name, returning its out_channels and out_strides.

        :param name: Name of the backbone to describe.
        :return: A dictionary containing the out_channels and out_strides of the backbone.
        """
        cfg = get_backbone_default_config(name)
        print(f"{name}:")
        for f in fields(cfg):
            current = getattr(cfg, f.name)
            print(f"  {f.name}: {f.type}  = {current!r}")

    @classmethod
    def config(cls, name: str) -> Any: # Factory method to get the default config of a backbone by name
        """Get the default configuration for a backbone by name.

        :param name: Name of the backbone to get the configuration for.
        :return: The default configuration of the backbone.
        """
        return get_backbone_default_config(name)

    @classmethod
    def list_backbones(cls, module: str | None = None, filter: str | None = None) -> list[str]: # Factory method to list all registered backbones
        """
        List all registered backbones, optionally filtered by module and/or a wildcard pattern.

        :param module: Optional module name to filter the backbones by.
        :param filter: Optional wildcard pattern to filter the backbones by.
        :return: A list of registered backbone names.
        """
        return list_backbones(module=module, filter=filter)

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        raise NotImplementedError
