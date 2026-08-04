from dataclasses import dataclass
from typing import Sequence
import torch.nn as nn

from .base import Head
from ._registry import register_head
from ..nn.blocks.convolution.conv_norm_act import ConvNormAct
from ..nn.features import FeatureMaps, FeatureSpec, HeadOutput


__all__ = ["MaskRCNNHead"]


@dataclass
class MaskRCNNHeadConfig:
    conv_dims: Sequence[int] = (256, 256, 256, 256)
    upsample_dim: int = 256
    num_classes: int = 80
    class_agnostic: bool = False


class MaskRCNNHead(Head):
    def __init__(self, in_spec: FeatureSpec, cfg: MaskRCNNHeadConfig):
        super().__init__()
        in_spec.require("channels")
        if "roi" not in in_spec.channels:
            raise ValueError("MaskRCNNHead requires 'roi' feature maps in the input spec.")

        cur_channels = in_spec.channels["roi"]
        convs: list[nn.Module] = []
        for conv_dim in cfg.conv_dims:
            convs.append(
                ConvNormAct(
                    in_channels=cur_channels,
                    out_channels=conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=None,
                    activation="relu",
                )
            )
            cur_channels = conv_dim

        self.conv_tower = nn.Sequential(*convs)
        self.deconv = nn.ConvTranspose2d(cur_channels, cfg.upsample_dim, kernel_size=2, stride=2)
        self.deconv_relu = nn.ReLU(inplace=True)
        out_channels = 1 if cfg.class_agnostic else cfg.num_classes
        self.predictor = nn.Conv2d(cfg.upsample_dim, out_channels, kernel_size=1, stride=1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: FeatureMaps) -> HeadOutput:
        if "roi" not in features.feature_maps:
            raise ValueError("MaskRCNNHead requires ROI feature maps under features.feature_maps['roi'].")

        x = features.feature_maps["roi"]
        x = self.conv_tower(x)
        x = self.deconv_relu(self.deconv(x))
        masks = self.predictor(x)
        return HeadOutput(masks=masks)


mask_head_configs = {
    "mask_rcnn_head": MaskRCNNHeadConfig(),
}


@register_head(config=mask_head_configs["mask_rcnn_head"])
def mask_rcnn_head(in_spec: FeatureSpec, cfg: MaskRCNNHeadConfig) -> MaskRCNNHead:
    return MaskRCNNHead(in_spec, cfg)
