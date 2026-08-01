from __future__ import annotations

from abc import ABC
from dataclasses import replace, fields
from typing import Any, Optional, Union
import torch.nn as nn

from ._registry import get_neck_entrypoint, get_neck_default_config, list_necks
from ..nn.features import FeatureMaps, FeatureSpec


__all__ = ["Neck"]


class Neck(nn.Module, ABC):
    """A neck only produces features -- it knows nothing about tasks."""

    out_spec: FeatureSpec

    @staticmethod
    def _validate_out_spec(neck: "Neck") -> None:
        """Ensure every created neck exposes a valid FeatureSpec."""
        if not isinstance(neck.out_spec, FeatureSpec):
            raise ValueError(
                f"{neck.__class__.__name__} must define an 'out_spec' attribute of type FeatureSpec. Check docs for details."
            )

    @classmethod
    def create(
            cls,
            name: str,
            *,
            backbone: Optional[Union[nn.Module, None]] = None,
            **overrides
        ) -> Neck:  # Factory method to create a neck by name
        """Create a neck by name, optionally loading pretrained weights.

        :param name: Name of the neck to create.
        :param backbone: Optional backbone module to infer the input feature specification from.
        If provided, the neck will use the backbone's output feature specification as its
        input feature specification. If not provided, the user must provide an 'in_spec'
        override in the keyword arguments.
        :param overrides: Optional keyword arguments to overwrite the default configuration.
        :return: An instance of the neck.
        """
        if backbone is not None and overrides.get("in_spec") is not None:
            raise ValueError("Do not provide 'in_spec' when a backbone is given.")
        
        if backbone is not None:
            in_spec = getattr(backbone, "out_spec", None)
            if in_spec is None:
                raise ValueError(
                    "The provided backbone does not have an 'out_spec' attribute."
                    " There is something wrong with the backbone implementation."
                )
            overrides["in_spec"] = in_spec
            
        if backbone is None:
            in_spec = overrides.get("in_spec")
            if in_spec is None:
                raise ValueError("Must provide 'in_spec' when no backbone is given.")

        if in_spec is not None and not isinstance(in_spec, FeatureSpec):
            raise TypeError(
                f"The provided 'in_spec' must be an instance of FeatureSpec, got {type(in_spec)} instead. "
                "Check the documentation for details."
            )

        # Get the entrypoint and default configuration for the specified neck
        entrypoint = get_neck_entrypoint(name)
        default_cfg = get_neck_default_config(name)

        # Replace overrides (raises on unknown fields)
        cfg = replace(default_cfg, **overrides)

        # Create the neck using the entrypoint and validate its out_spec
        neck = entrypoint(cfg)
        cls._validate_out_spec(neck)
        return neck

    @classmethod
    def describe(cls, name: str) -> dict[str, int]: # Factory method to describe a neck by name
        """
        Describe a neck by name, returning its out_channels and out_strides.

        :param name: Name of the neck to describe.
        :return: A dictionary containing the out_channels and out_strides of the neck.
        """
        cfg = get_neck_default_config(name)
        print(f"{name}:")
        for f in fields(cfg):
            current = getattr(cfg, f.name)
            print(f"  {f.name}: {f.type}  = {current!r}")

    @classmethod
    def config(cls, name: str) -> Any: # Factory method to get the default config of a backbone by name
        """Get the default configuration for a neck by name.

        :param name: Name of the neck to get the configuration for.
        :return: The default configuration of the neck.
        """
        return get_neck_default_config(name)

    @classmethod
    def list_necks(cls, module: str | None = None, filter: str | None = None) -> list[str]: # Factory method to list all registered backbones
        """
        List all registered necks, optionally filtered by module and/or a wildcard pattern.

        :param module: Optional module name to filter the necks by.
        :param filter: Optional wildcard pattern to filter the necks by.
        :return: A list of registered neck names.
        """
        return list_necks(module=module, filter=filter)
    
    def forward(self, features: FeatureMaps) -> FeatureMaps:
        raise NotImplementedError
    