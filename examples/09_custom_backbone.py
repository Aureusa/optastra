"""Write your own backbone from the reusable blocks in `optastra.nn.blocks`,
register a family of variants in a loop, and plug it into everything that
already exists -- pooling necks, heads, FPN -- without touching any of them.

What it shows:
    1. The Backbone contract: take images, return `FeatureMaps`, and describe
       the output up front in `self.out_spec` (a `FeatureSpec`: channels and
       stride per feature level). Downstream components are wired from the
       spec at construction time, never by guessing at forward time.
    2. "Ideas, not papers": the network is ConvNormAct + ResidualBlock +
       SqueezeExcitation -- blocks the built-in backbones use too.
    3. `Backbone.register(..., name=...)` in a loop registers S/M/L variants
       from one table; `Backbone.create("mininet_m", use_se=False)` overrides
       any config field.
    4. The FeatureSpec catches wiring mistakes when the model is built:
       a classification head on a spatial backbone without a pooling neck.

Run with:
    python examples/09_custom_backbone.py
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from optastra import Backbone, Head, Neck, build_sequential_model
from optastra.nn.blocks.convolution import ConvNormAct, ResidualBlock, SqueezeExcitation
from optastra.nn.features import FeatureMaps, FeatureSpec


# --- 1. Config + component --------------------------------------------------

@dataclass
class MiniNetConfig:
    in_channels: int = 3
    widths: list[int] = field(default_factory=lambda: [32, 64, 128])   # one stage per entry
    blocks_per_stage: int = 1
    use_se: bool = True                                               # squeeze-excitation after each stage


class MiniNet(Backbone):
    """stem (stride 2) -> stages (each halves the resolution) -> {"C2", "C3", ...}.

    Stage i ends at stride 2**(i+2), so it is named "C{i+2}" -- the same
    convention as the built-in CNNs, which is what lets FPN, detectors and
    the pooling necks consume it unchanged.
    """

    def __init__(self, cfg: MiniNetConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = ConvNormAct(in_channels=cfg.in_channels, out_channels=cfg.widths[0], kernel_size=3, stride=2)

        self.stages = nn.ModuleList()
        in_ch = cfg.widths[0]
        for width in cfg.widths:
            blocks = [ResidualBlock(in_ch, width, stride=2)]
            blocks += [ResidualBlock(width, width) for _ in range(cfg.blocks_per_stage - 1)]
            if cfg.use_se:
                blocks.append(SqueezeExcitation(width))
            self.stages.append(nn.Sequential(*blocks))
            in_ch = width

        self.stage_names = [f"C{i + 2}" for i in range(len(cfg.widths))]
        # Declared once, checked by every consumer at construction time.
        self.out_spec = FeatureSpec(
            channels={name: w for name, w in zip(self.stage_names, cfg.widths)},
            strides={name: 2 ** (i + 2) for i, name in enumerate(self.stage_names)},
        )

    def forward(self, images: torch.Tensor) -> FeatureMaps:
        x = self.stem(images)
        feature_maps = {}
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            feature_maps[name] = x
        return FeatureMaps(feature_maps=feature_maps)


# --- 2. Register a table of variants in a loop ------------------------------

MININET_VARIANTS = {
    "mininet_s": MiniNetConfig(widths=[16, 32, 64]),
    "mininet_m": MiniNetConfig(widths=[32, 64, 128], blocks_per_stage=2),
    "mininet_l": MiniNetConfig(widths=[64, 128, 256, 512], blocks_per_stage=2),
}


def mininet(cfg: MiniNetConfig) -> MiniNet:
    return MiniNet(cfg)


for variant, variant_cfg in MININET_VARIANTS.items():
    Backbone.register(mininet, config=variant_cfg, name=variant)


def main() -> None:
    torch.manual_seed(0)
    print("registered:", Backbone.list_all(filter="mininet"))
    Backbone.describe("mininet_m")

    # --- 3. The spec matches what forward() actually produces ----------------
    backbone = Backbone.create("mininet_m", use_se=False)   # any config field can be overridden
    feats = backbone(torch.randn(2, 3, 64, 64))
    for name, fmap in feats.feature_maps.items():
        stride = 64 // fmap.shape[-1]
        print(f"  {name}: {tuple(fmap.shape)}  spec: {backbone.out_spec.channels[name]} ch, "
              f"stride {backbone.out_spec.strides[name]} (measured {stride})")

    # --- 4. Plug into existing components -------------------------------------
    classifier = build_sequential_model(
        backbone="mininet_s", necks=["global_avg_pool"],
        head=("vanilla_classification_head", {"num_classes": 10, "hidden_features": 64}),
    )
    print("classifier logits:", tuple(classifier(torch.randn(2, 3, 64, 64)).logits.shape))

    fpn = Neck.create("fpn", in_spec=backbone.out_spec, out_channels=64)
    pyramid = fpn(feats)
    print("FPN on MiniNet:", {k: tuple(v.shape[1:]) for k, v in pyramid.feature_maps.items()},
          "strides", fpn.out_spec.strides)

    # --- 5. Wiring mistakes fail at construction, with a readable message ----
    try:
        Head.create("vanilla_classification_head", in_spec=backbone.out_spec, num_classes=10)
    except ValueError as err:
        print("head without a pooling neck ->", str(err).splitlines()[0])


if __name__ == "__main__":
    main()
