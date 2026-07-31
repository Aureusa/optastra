"""
This module implements a convolutional layer followed by an optional normalization and activation layer.
The sequence of operations is Conv -> Norm -> Act, which is a common pattern in many CNN
architectures.
"""
from dataclasses import dataclass
import torch.nn as nn


_NORMS = {
    "batchnorm": nn.BatchNorm2d,
    "layernorm": nn.LayerNorm,
    "groupnorm": nn.GroupNorm,
    None: None
}
_ACTS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    None: None
}


@dataclass
class ConvNormActConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    groups: int = 1
    padding: int | None = None
    norm: str | None = "batchnorm"
    activation: str | None = "relu"
    bias: bool | None = None


class ConvNormAct(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        padding: int | None = None,
        norm: str | None = "batchnorm",
        activation: str | None = "relu",
        bias: bool | None = None,
        preact: bool = False
    ):
        """
        Implements a convolutional layer followed by an optional
        normalization and activation layer.
        Conv -> Norm -> Act
        or with pre-activation:
        Norm -> Act -> Conv
        This operation is a primitive operation in many CNN architectures.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolutional kernel.
        :param stride: Stride of the convolution. Default is 1.
        :param groups: Number of blocked connections from input channels to output channels. Default is 1.
        :param padding: Padding added to all four sides of the input.
        If None, it defaults to kernel_size // 2 (same padding). Default is None.
        :param norm: Type of normalization to apply.
        Options are "batchnorm", "layernorm", "groupnorm", or None. Default is "batchnorm".
        :param activation: Type of activation function to apply.
        Options are "relu", "gelu", or None. Default is "relu".
        :param bias: If True, adds a learnable bias to the output.
        If None, it defaults to True if norm is None, otherwise False. Default is None.
        """
        super(ConvNormAct, self).__init__()
        if bias is None:
            bias = norm is None  # skip bias if norm follows (norm absorbs it)

        if padding is None:
            padding = kernel_size // 2  # default to same padding
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            groups=groups, bias=bias,
        )
        self.norm = _NORMS[norm](out_channels) if norm else nn.Identity()
        self.act = _ACTS[activation]() if activation else nn.Identity()

        self.preact = preact

    @classmethod
    def from_config(cls, config: ConvNormActConfig):
        return cls(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            groups=config.groups,
            padding=config.padding,
            norm=config.norm,
            activation=config.activation,
            bias=config.bias,
        )

    def forward(self, x):
        if self.preact:
            x = self.norm(x)
            x = self.act(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
        return x