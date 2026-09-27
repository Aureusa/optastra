"""Compatibility re-export: StochasticDepth now lives in optastra.nn.layers."""
from ...layers.stochastic_depth import StochasticDepth, drop_path_rates

__all__ = ["StochasticDepth", "drop_path_rates"]
