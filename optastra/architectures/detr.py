"""DETR-family set predictors: Backbone -> (Neck) -> (Encoder) -> Decoder -> Head."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..backbones.base import Backbone
from ..core.component_ref import ComponentRef, ComponentRefConfigMixin, component_field
from ..decoders import keys as decoder_keys
from ..decoders.base import Decoder, QuerySet
from ..detection import keys
from ..heads.base import Head
from ..necks.base import Neck
from ..nn.features import FeatureMaps, HeadOutput
from .base import Architecture, resolve_image_sizes


__all__ = [
    "DETR", "DETRConfig",
    "detr_r50", "conditional_detr_r50", "dab_detr_r50", "point_detr_r18_fpn",
]


@dataclass
class DETRConfig(ComponentRefConfigMixin):
    backbone: ComponentRef = component_field(Backbone, default_name="resnet50")
    neck: ComponentRef | None = component_field(Neck, optional=True)                 # e.g. fpn
    encoder: ComponentRef | None = component_field(Neck, default_name="detr_encoder")   # None = no encoder
    decoder: ComponentRef = component_field(Decoder, default_name="detr_decoder")
    head: ComponentRef = component_field(Head, default_name="detr_box_head")
    d_model: int = 256                 # injected into the encoder and decoder
    num_classes: int | None = 80       # injected into the head when set (None for heads without classes)
    aux_loss: bool = True              # run the head on every decoder layer


def _resolve(ref: ComponentRef, factory, field: str, value, **extras):
    if field in ref.overrides and ref.overrides[field] != value:
        raise ValueError(f"{ref.name}: {field}={ref.overrides[field]} conflicts with the architecture's "
                         f"{field}={value}; set it on the architecture instead.")
    overrides = {k: v for k, v in ref.overrides.items() if k != field}
    return factory.create(ref.name, **overrides, **{field: value}, **extras)


class DETR(Architecture):
    """DETR (Carion et al., 2020) and its query variants, as a composition.

    Every part is a ComponentRef: any backbone, an optional neck (e.g. FPN for
    a finer stride), an optional ``detr_encoder``, a ``Decoder`` (vanilla /
    conditional / DAB queries via the decoder's ``query_type``) and a head on
    the decoder tokens -- ``detr_box_head`` for boxes, ``keypoint_head`` for
    points. The head runs on every decoder layer; the earlier layers' outputs
    are in ``extra["aux_outputs"]`` for the auxiliary losses.

    ``forward(images, image_sizes=None, queries=None)`` -> HeadOutput of the
    last layer, with ``extra`` = image_sizes, query_mask, aux_outputs,
    reference_points (if any) and the decoder tokens (``keys.DECODER_TOKENS``).
    """

    def __init__(self, cfg: DETRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = cfg.backbone.resolve(Backbone)
        spec = self.backbone.out_spec
        self.neck = cfg.neck.resolve(Neck, in_spec=spec) if cfg.neck is not None else None
        if self.neck is not None:
            spec = self.neck.out_spec
        self.encoder = _resolve(cfg.encoder, Neck, "d_model", cfg.d_model, in_spec=spec) if cfg.encoder is not None else None
        if self.encoder is not None:
            spec = self.encoder.out_spec
        self.decoder = _resolve(cfg.decoder, Decoder, "d_model", cfg.d_model, in_spec=spec)
        head_extras = {"num_classes": cfg.num_classes} if cfg.num_classes is not None else {}
        self.head = cfg.head.resolve(Head, in_spec=self.decoder.out_spec, **head_extras)

    def forward(
        self,
        images: torch.Tensor,
        image_sizes: list[tuple[int, int]] | torch.Tensor | None = None,
        queries: QuerySet | None = None,
        gt_boxes=None,
    ) -> HeadOutput:
        """
        :param images: (B, C, H, W), possibly padded bottom / right.
        :param image_sizes: per-image (h, w) before padding.
        :param queries: optional external queries (needs a decoder with reference points).
        :param gt_boxes: accepted for DetectionTask compatibility and ignored (no proposals to augment).
        """
        image_sizes = resolve_image_sizes(images, image_sizes)
        features = self.backbone(images)
        if self.neck is not None:
            features = self.neck(features)
        features.extra[keys.IMAGE_SIZES] = image_sizes
        if self.encoder is not None:
            features = self.encoder(features)
        decoded = self.decoder(features, queries)
        layers = decoded.extra[decoder_keys.LAYER_TOKENS] if self.cfg.aux_loss else [decoded.patch_tokens]
        outputs = [self.head(FeatureMaps(patch_tokens=tokens, extra=decoded.extra)) for tokens in layers]
        final = outputs[-1]
        final.extra.update({
            keys.AUX_OUTPUTS: outputs[:-1],
            keys.IMAGE_SIZES: image_sizes,
            keys.QUERY_MASK: decoded.extra[decoder_keys.QUERY_MASK],
            keys.DECODER_TOKENS: decoded.patch_tokens,
        })
        if decoder_keys.REFERENCE_POINTS in decoded.extra:
            final.extra[decoder_keys.REFERENCE_POINTS] = decoded.extra[decoder_keys.REFERENCE_POINTS]
        return final


detr_configs = {
    # DETR-R50: encoder on C5 (stride 32), 100 vanilla queries, 80 COCO classes.
    "detr_r50": DETRConfig(),
    "conditional_detr_r50": DETRConfig(decoder=ComponentRef("conditional_detr_decoder")),
    "dab_detr_r50": DETRConfig(decoder=ComponentRef("dab_detr_decoder")),
    # Point detection of small sources: FPN on C3-C5 and a multi-scale encoder over P3-P5
    # (finest stride 8 -- small sources vanish at DETR's stride 32), keypoint head.
    "point_detr_r18_fpn": DETRConfig(
        backbone=ComponentRef("resnet18"),
        neck=ComponentRef("fpn", {"in_features": ("C3", "C4", "C5")}),
        encoder=ComponentRef("detr_encoder", {"in_features": ("P3", "P4", "P5")}),
        head=ComponentRef("keypoint_head"),
        num_classes=None,
    ),
}


@Architecture.register(config=detr_configs["detr_r50"])
def detr_r50(cfg: DETRConfig) -> DETR:
    return DETR(cfg)


@Architecture.register(config=detr_configs["conditional_detr_r50"])
def conditional_detr_r50(cfg: DETRConfig) -> DETR:
    return DETR(cfg)


@Architecture.register(config=detr_configs["dab_detr_r50"])
def dab_detr_r50(cfg: DETRConfig) -> DETR:
    return DETR(cfg)


@Architecture.register(config=detr_configs["point_detr_r18_fpn"])
def point_detr_r18_fpn(cfg: DETRConfig) -> DETR:
    return DETR(cfg)
