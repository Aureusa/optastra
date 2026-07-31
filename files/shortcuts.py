from __future__ import annotations

from vision.nn.convolutions.conv_norm_act import ConvNormAct


class ShortcutProjection(ConvNormAct):
    """1x1 conv + BN, no activation. Used whenever a residual branch changes
    channels or spatial resolution (He et al. 2015, 'option B' projection shortcut).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            norm="batchnorm",
            activation=None,
        )
