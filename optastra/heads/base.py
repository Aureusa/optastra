from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace, fields

import torch
import torch.nn as nn
from typing import Any, Optional, Union

from ._registry import (
    get_head_entrypoint,
    get_head_default_config,
    list_heads,
    check_head_registered
)
from ..nn.features import FeatureMaps, FeatureSpec, HeadOutput


__all__ = ["Head"]


class Head(nn.Module, ABC):
    """A head only produces features -- it knows nothing about tasks."""

    @staticmethod
    def _validate_in_spec(in_spec: Any) -> None:
        if not isinstance(in_spec, FeatureSpec):
            raise TypeError(f"'in_spec' must be a FeatureSpec, got {type(in_spec)}.")

    @classmethod
    def create(
            cls,
            name: str,
            in_spec: FeatureSpec,
            **overrides
        ) -> Head:  # Factory method to create a head by name
        if not check_head_registered(name):  # Ensure the head is registered
            raise ValueError(f"Head '{name}' is not registered.")

        # Validate the provided in_spec
        cls._validate_in_spec(in_spec)

        # Get the entrypoint and default configuration for the specified head
        entrypoint = get_head_entrypoint(name)
        default_cfg = get_head_default_config(name)

        # Replace overrides (raises on unknown fields)
        cfg = replace(default_cfg, **overrides)

        # Create the head using the entrypoint
        head = entrypoint(in_spec, cfg)
        return head

    @classmethod
    def describe(cls, name: str) -> dict[str, int]: # Factory method to describe a head by name
        """
        Describe a head by name, returning its out_channels and out_strides.

        :param name: Name of the head to describe.
        :return: A dictionary containing the out_channels and out_strides of the head.
        """
        cfg = get_head_default_config(name)
        print(f"{name}:")
        for f in fields(cfg):
            current = getattr(cfg, f.name)
            print(f"  {f.name}: {f.type}  = {current!r}")

    @classmethod
    def config(cls, name: str) -> Any: # Factory method to get the default config of a backbone by name
        """Get the default configuration for a head by name.

        :param name: Name of the head to get the configuration for.
        :return: The default configuration of the head.
        """
        return get_head_default_config(name)

    @classmethod
    def list_heads(cls, module: str | None = None, filter: str | None = None) -> list[str]: # Factory method to list all registered heads
        """
        List all registered heads, optionally filtered by module and/or a wildcard pattern.

        :param module: Optional module name to filter the heads by.
        :param filter: Optional wildcard pattern to filter the heads by.
        :return: A list of registered head names.
        """
        return list_heads(module=module, filter=filter)
    
    def forward(self, features: FeatureMaps) -> HeadOutput:
        raise NotImplementedError
    