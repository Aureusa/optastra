"""
This module contains the implementation of residual blocks in PyTorch.
It implements a few main ideas:
    - the basic residual block (Reference: He et al. 2015)
    - the bottleneck residual block (Reference: He et al. 2015)
    - the downsampled residual block (Reference: He et al. 2015)
    - the pre-activation residual block (Reference: He et al. 2016)
"""
import torch
import torch.nn as nn

from .conv_norm_act import ConvNormAct


def _make_downsampling_layer(
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int,
        preact: bool = False
    ) -> nn.Module:
    """
    Creates a downsample layer for a residual block.
    It checks if a projection shortcut is needed based on the input
    and output channels and the stride. If a projection is needed,
    it returns a ShortcutProjection layer; otherwise, it returns None.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param stride: Stride for the convolutional layer.
    :param expansion: Expansion factor for the block (1 for BasicBlock, 4 for BottleneckBlock).
    :param preact: Whether to use pre-activation. Default is False.
    :return: A downsample layer (ShortcutProjection) or None.
    """
    need_projection = stride != 1 or in_channels != out_channels * expansion
    if need_projection:
        return ShortcutProjection(in_channels, out_channels * expansion, stride=stride, preact=preact)
    return None


class ShortcutProjection(ConvNormAct):
    """1x1 conv + BN, no activation. Used whenever a residual branch changes
    channels or spatial resolution (He et al. 2015, 'option B' projection shortcut).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, preact: bool = False):
        """
        Initializes the ShortcutProjection layer.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the convolutional layer. Default is 1.
        :param preact: Whether to use pre-activation. Default is False.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            norm="batchnorm",
            activation=None,
            bias=False,
            preact=preact
        )


class ResidualBlock(nn.Module): # (ResNet-18/34)
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            preact: bool = False
        ):
        """
        Initializes the Residual Block. This is the basic building block for
        ResNet-18 and ResNet-34 architectures.
        Conv -> Norm -> Act -> Conv -> Norm -> Add -> Act
        or with pre-activation:
        Norm -> Act -> Conv -> Norm -> Act -> Conv -> Add

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the first convolutional layer. Default is 1.
        :param preact: Whether to use pre-activation. Default is False.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm="batchnorm",
            activation="relu",
            preact=preact
        )
        self.conv2 = ConvNormAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm="batchnorm",
            activation=None if not preact else "relu",
            preact=preact
        )

        self.downsample = _make_downsampling_layer(
            in_channels, out_channels, stride, self.expansion, preact=preact
        )

        self.act = nn.ReLU(inplace=True)
        self.preact = preact

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.preact:
            out = self.act(out)
        return out


class BottleneckResidualBlock(nn.Module): # (ResNet-50/101/152)
    expansion = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            preact: bool = False
        ):
        """
        Initializes the Bottleneck Residual Block. This is the building block for
        ResNet-50, ResNet-101, and ResNet-152 architectures.
        Conv -> Norm -> Act -> Conv -> Norm -> Act -> Conv -> Norm -> Add -> Act
        or with pre-activation:
        Norm -> Act -> Conv -> Norm -> Act -> Conv -> Norm -> Act -> Conv -> Add

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the second convolutional layer. Default is 1.
        :param preact: Whether to use pre-activation. Default is False.
        """
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm="batchnorm",
            activation="relu",
            preact=preact
        )
        self.conv2 = ConvNormAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm="batchnorm",
            activation="relu",
            preact=preact
        )
        self.conv3 = ConvNormAct(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            norm="batchnorm",
            activation=None if not preact else "relu",
            preact=preact
        )

        self.downsample = _make_downsampling_layer(
            in_channels, out_channels, stride, self.expansion, preact=preact
        )

        self.preact = preact
        if not preact:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if not self.preact:
            out = self.act(out)
        return out


class DownsampleResidualBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 2,
            preact: bool = False
        ):
        """
        Initializes the Downsample Residual Block. This block is used to reduce
        the spatial dimensions of the input while increasing the number of channels.
        Conv -> Norm -> Act -> Conv -> Norm -> Add -> Act
        or with pre-activation:
        Norm -> Act -> Conv -> Norm -> Act -> Conv -> Add

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the first convolutional layer. Default is 2.
        """
        super(DownsampleResidualBlock, self).__init__()
        self.conv1 = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm="batchnorm",
            activation="relu",
            preact=preact
        )
        self.conv2 = ConvNormAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm="batchnorm",
            activation=None if not preact else "relu",
            preact=preact
        )
        self.downsample = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            norm="batchnorm",
            activation=None if not preact else "relu",
            preact=preact
        )

        self.preact = preact
        if not preact:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        identity = self.downsample(x)

        out += identity
        if not self.preact:
            out = self.act(out)
        return out
    