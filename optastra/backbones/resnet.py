"""
ResNet backbone implementation. It adopts the design of ResNet from the original
paper "Deep Residual Learning for Image Recognition" by Kaiming He et al. (2015).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .base import Backbone
from ..nn.blocks.convolution.residual import ResidualBlock, BottleneckResidualBlock
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.features import FeatureSpec, FeatureMaps


__all__ = ["ResNet", "ResNetConfig", "RESNET_BLOCKS"]


# Config value -> block class. Configs store the string so they stay YAML-safe.
RESNET_BLOCKS: dict[str, type[ResidualBlock | BottleneckResidualBlock]] = {
    "basic": ResidualBlock,               # 2x (3x3 conv), expansion 1 -- ResNet-18/34
    "bottleneck": BottleneckResidualBlock,  # 1x1 -> 3x3 -> 1x1, expansion 4 -- ResNet-50/101/152
}


@dataclass
class ResNetConfig:
    """Config for the ResNet family."""
    block: str = field(default="basic", metadata={"choices": list(RESNET_BLOCKS)})
    layers: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    in_channels: int = 3
    stem_channels: int = 64
    preact: bool = False
    zero_init_residual: bool = False  # zero the last BN gamma of each residual branch (Goyal et al. 2017)


class ResNetStem(nn.Module):
    """7x7 conv stride 2 -> BN -> ReLU -> 3x3 maxpool stride 2. Output stride 4 (this is C1)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.conv = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            norm="batchnorm",
            activation="relu",
            preact=False,
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.conv(x))
    

class ResNet(Backbone):
    """Generic ResNet. Returns multi-stage features C2-C5 for use in necks like FPN.

    Stage strides relative to input: C1 (stem) = 4, C2 = 4, C3 = 8, C4 = 16, C5 = 32.
    """

    def __init__(
        self,
        cfg: ResNetConfig,
    ):
        """
        Initializes the ResNet backbone.

        :param cfg: ResNetConfig. `block` is "basic" or "bottleneck" (see RESNET_BLOCKS),
            `layers` the number of blocks per stage, `preact` switches to the
            pre-activation variant (He et al. 2016), `zero_init_residual` starts every
            residual branch at zero so each block initially behaves like an identity.
        """
        super().__init__()
        self.cfg = cfg

        if cfg.block not in RESNET_BLOCKS:
            raise ValueError(f"Unknown ResNet block '{cfg.block}', expected one of {list(RESNET_BLOCKS)}.")

        # Unpack configuration parameters
        block = RESNET_BLOCKS[cfg.block]
        layers = cfg.layers
        in_channels = cfg.in_channels
        stem_channels = cfg.stem_channels
        preact = cfg.preact

        # Create the stem of the ResNet
        self.stem = ResNetStem(in_channels, stem_channels)

        stage_channels = [64, 128, 256, 512]
        stage_strides = [1, 2, 2, 2]  # stage1 keeps stride (pool already halved it)

        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for width, depth, stride in zip(stage_channels, layers, stage_strides):
            self.stages.append(self._make_stage(block, in_ch, width, depth, stride, preact=preact))
            in_ch = width * block.expansion

        # Pre-activation blocks end with a bare addition, so the paper adds a final
        # BN-ReLU after the last stage (He et al. 2016, Fig. 1b / Sec. 4).
        if preact:
            self.final_norm = nn.BatchNorm2d(in_ch)
            self.final_act = nn.ReLU(inplace=True)

        self.out_spec = FeatureSpec(
            channels={f"C{i + 2}": stage_channels[i] * block.expansion for i in range(4)},
            strides={"C2": 4, "C3": 8, "C4": 16, "C5": 32},
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """He (kaiming) normal init for convs, BN to identity; optionally zero-init
        the last BN of every residual branch (torchvision's `zero_init_residual`)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # In the pre-activation variant the branch ends with a conv, not a norm, so
        # there is no BN gamma to zero -- the option only applies to the post-act blocks.
        if self.cfg.zero_init_residual and not self.cfg.preact:
            for m in self.modules():
                if isinstance(m, (ResidualBlock, BottleneckResidualBlock)):
                    nn.init.zeros_(m.last_norm.weight)

    @staticmethod
    def _make_stage(
            block: type[ResidualBlock | BottleneckResidualBlock],
            in_channels: int,
            out_channels: int,
            depth: int,
            stride: int,
            preact: bool = False
        ) -> nn.Sequential:
        """
        Creates a stage of the ResNet architecture consisting of multiple residual blocks.
        
        :param block: The type of residual block to use (ResidualBlock or BottleneckResidualBlock).
        :param in_channels: Number of input channels to the stage.
        :param out_channels: Number of output channels for the blocks in the stage.
        :param depth: Number of residual blocks in the stage.
        :param stride: Stride for the first block in the stage. Subsequent blocks will have a stride of 1.
        :return: A sequential container of residual blocks forming the stage
        """
        layers = [block(in_channels, out_channels, stride=stride, preact=preact)]
        for _ in range(depth - 1):
            layers.append(block(out_channels * block.expansion, out_channels, preact=preact))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        """
        Forward pass through the ResNet backbone.
        
        :param images: Input tensor of shape (B, C, H, W).
        :return: FeatureMaps containing feature maps from each stage.
        """
        x = self.stem(images)
        feature_maps = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.cfg.preact and i == len(self.stages) - 1:
                x = self.final_act(self.final_norm(x))
            feature_maps[f"C{i + 2}"] = x
        return FeatureMaps(feature_maps=feature_maps)


resnet_configs = {
    "resnet18": ResNetConfig(block="basic", layers=[2, 2, 2, 2]),
    "resnet34": ResNetConfig(block="basic", layers=[3, 4, 6, 3]),
    "resnet50": ResNetConfig(block="bottleneck", layers=[3, 4, 6, 3]),
    "resnet101": ResNetConfig(block="bottleneck", layers=[3, 4, 23, 3]),
    "resnet152": ResNetConfig(block="bottleneck", layers=[3, 8, 36, 3]),
}


@Backbone.register(config=resnet_configs["resnet18"])
def resnet18(cfg: ResNetConfig) -> ResNet:
    return ResNet(cfg)

@Backbone.register(config=resnet_configs["resnet34"])
def resnet34(cfg: ResNetConfig) -> ResNet:
    return ResNet(cfg)

@Backbone.register(config=resnet_configs["resnet50"])
def resnet50(cfg: ResNetConfig) -> ResNet:
    return ResNet(cfg)

@Backbone.register(config=resnet_configs["resnet101"])
def resnet101(cfg: ResNetConfig) -> ResNet:
    return ResNet(cfg)

@Backbone.register(config=resnet_configs["resnet152"])
def resnet152(cfg: ResNetConfig) -> ResNet:
    return ResNet(cfg)
