from __future__ import annotations

import torch.nn as nn

_NORMS = {"batchnorm": nn.BatchNorm2d, None: None}
_ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, None: None}


class ConvNormAct(nn.Module):
    """conv -> norm -> activation, the atomic primitive most blocks are built from."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        norm: str | None = "batchnorm",
        activation: str | None = "relu",
        bias: bool | None = None,
    ):
        super().__init__()
        if bias is None:
            bias = norm is None  # skip bias if a norm layer follows (it absorbs it)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
        )
        self.norm = _NORMS[norm](out_channels) if norm else nn.Identity()
        self.act = _ACTS[activation]() if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
