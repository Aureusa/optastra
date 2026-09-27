"""
Shared low-level layers used by blocks of every kind (convolutional,
transformer, ...). Anything here is a small, self-contained ``nn.Module``
or helper with no dependency on ``optastra.nn.blocks``.
"""
from .layernorm2d import LayerNorm2d
from .stochastic_depth import StochasticDepth, drop_path_rates

__all__ = ["LayerNorm2d", "StochasticDepth", "drop_path_rates"]
