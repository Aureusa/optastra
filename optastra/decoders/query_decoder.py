"""DETR's query decoder, with vanilla, Conditional-DETR and DAB-DETR queries."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..nn.blocks.readout.mlp import MLP
from ..nn.blocks.transformer.detr_layers import DETRDecoderLayer
from ..nn.blocks.transformer.memory import flatten_feature_levels
from ..nn.blocks.transformer.pos_embed import sine_embed_2d
from ..nn.features import FeatureMaps, FeatureSpec
from . import keys
from .base import Decoder, QuerySet


__all__ = [
    "QueryDecoder", "QueryDecoderConfig", "QUERY_TYPES", "inverse_sigmoid",
    "detr_decoder", "conditional_detr_decoder", "dab_detr_decoder",
]

QUERY_TYPES = ("vanilla", "conditional", "dab")


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


@dataclass
class QueryDecoderConfig:
    d_model: int = 256
    num_queries: int = 100            # learned queries (external queries come on top)
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    normalize_before: bool = False    # DETR is post-norm
    query_type: str = "vanilla"       # "vanilla" | "conditional" | "dab"
    in_features: tuple[str, ...] = ()  # feature maps used as memory; () = all maps in the in_spec
    external_queries: bool = False    # conditional only: build the MLP that turns external reference points
                                      # into positional queries (dab and vanilla need no extra parameters)


class QueryDecoder(Decoder):
    """M object queries cross-attend to the image features and become object tokens.

    ``query_type`` selects how a query carries position (STARLING.md's
    ablation axis; Carion et al. 2020, Meng et al. 2021, Liu et al. 2022):

    * ``"vanilla"`` (DETR) -- a learned positional embedding per query, added to
      the queries of every attention. Queries have no reference point; heads
      predict absolute positions.
    * ``"conditional"`` (Conditional DETR) -- each learned embedding also
      predicts a reference point; cross-attention concatenates a content part
      with a spatial part (the sine embedding of the reference point, scaled
      by a function of the content), so attention is steered to the query's
      neighbourhood. Heads predict offsets from the reference point.
    * ``"dab"`` (DAB-DETR) -- the reference point *is* the learned query (an
      anchor); its sine embedding through an MLP gives the positional query.
      Same conditional cross-attention. Reference points stay fixed across
      layers here (no iterative anchor refinement).

    External queries (``forward(features, queries=QuerySet(...))``) are put in
    front of the learned ones -- e.g. priors at known source positions. They
    need reference points, so they are not available with vanilla queries
    (and need ``external_queries=True`` with conditional ones).

    Content starts at zero for learned queries (as in DETR), so the first
    layer's self-attention value projection gets no gradient unless external
    queries bring content -- harmless (it is still used) and the same as DETR.
    Every layer's output, after the shared final LayerNorm, is returned for
    auxiliary losses. ``out_spec.num_tokens`` counts the learned queries only;
    with K external queries the output has K + num_queries tokens.
    """

    def __init__(self, in_spec: FeatureSpec, cfg: QueryDecoderConfig):
        super().__init__()
        in_spec.require("channels", "strides")
        if cfg.query_type not in QUERY_TYPES:
            raise ValueError(f"query_type must be one of {QUERY_TYPES}, got {cfg.query_type!r}.")
        if cfg.d_model % 4 != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by 4 (2D sine position encodings).")
        self.cfg = cfg
        D = cfg.d_model
        names = list(cfg.in_features) or sorted(in_spec.channels, key=lambda n: in_spec.strides[n])
        missing = [n for n in names if n not in in_spec.channels or n not in in_spec.strides]
        if missing:
            raise ValueError(f"QueryDecoder: features {missing} not in the in_spec ({list(in_spec.channels)}).")
        self.feature_names = names
        self.strides = [in_spec.strides[n] for n in names]
        self.input_proj = nn.ModuleDict({
            n: nn.Identity() if in_spec.channels[n] == D else nn.Conv2d(in_spec.channels[n], D, 1) for n in names
        })
        self.level_embed = nn.Parameter(torch.randn(len(names), D) * 0.02) if len(names) > 1 else None

        M = cfg.num_queries
        if cfg.query_type == "vanilla":
            self.query_embed = nn.Embedding(M, D)
        elif cfg.query_type == "conditional":
            self.query_embed = nn.Embedding(M, D)
            self.ref_point_head = MLP(D, D, 2, num_layers=2)          # query_pos -> reference point
            if cfg.external_queries:
                self.pos_from_ref = MLP(D, D, D, num_layers=2)        # external queries: reference -> query_pos
        else:
            self.anchors = nn.Parameter(inverse_sigmoid(torch.rand(M, 2)))
            self.ref_point_head = MLP(D, D, D, num_layers=2)          # sine(reference) -> query_pos
        if cfg.query_type != "vanilla":
            self.query_scale = MLP(D, D, D, num_layers=2)             # content -> scale of the spatial query
        cross = "standard" if cfg.query_type == "vanilla" else "conditional"
        self.layers = nn.ModuleList(
            DETRDecoderLayer(D, cfg.num_heads, cfg.dim_feedforward, cfg.dropout, cfg.normalize_before, cross,
                             first_layer=(i == 0))
            for i in range(cfg.num_layers)
        )
        self.norm = nn.LayerNorm(D)
        self.out_spec = FeatureSpec(embed_dim=D, num_tokens=M)

    @property
    def uses_reference_points(self) -> bool:
        return self.cfg.query_type != "vanilla"

    def _learned_queries(self, B: int, device, dtype):
        D, M = self.cfg.d_model, self.cfg.num_queries
        tgt = torch.zeros(B, M, D, device=device, dtype=dtype)
        if self.cfg.query_type == "vanilla":
            return tgt, self.query_embed.weight.to(dtype).expand(B, M, D), None
        if self.cfg.query_type == "conditional":
            query_pos = self.query_embed.weight.to(dtype).expand(B, M, D)
            return tgt, query_pos, self.ref_point_head(query_pos).sigmoid().float()
        return tgt, None, self.anchors.sigmoid().float().expand(B, M, 2)

    def _external_queries(self, queries: QuerySet, B: int, device, dtype):
        if not self.uses_reference_points:
            raise ValueError("Vanilla DETR queries have no reference points, so external queries cannot be "
                             "placed; use query_type='conditional' or 'dab'.")
        if queries.reference_points is None:
            raise ValueError("External queries need reference_points.")
        if self.cfg.query_type == "conditional" and not self.cfg.external_queries:
            raise ValueError("Conditional queries take external queries only with external_queries=True "
                             "(it adds the reference -> position MLP they need).")
        K, D = queries.reference_points.shape[1], self.cfg.d_model
        tgt = queries.content.to(dtype) if queries.content is not None else torch.zeros(B, K, D, device=device, dtype=dtype)
        refs = queries.reference_points.float()
        query_pos = self.pos_from_ref(sine_embed_2d(refs, D).to(dtype)) if self.cfg.query_type == "conditional" else None
        mask = queries.mask if queries.mask is not None else torch.ones(B, K, dtype=torch.bool, device=device)
        return tgt, query_pos, refs, mask

    def forward(self, features: FeatureMaps, queries: QuerySet | None = None) -> FeatureMaps:
        maps = [self.input_proj[n](features.feature_maps[n]) for n in self.feature_names]
        image_sizes = features.extra.get(keys.IMAGE_SIZES)
        flat = flatten_feature_levels(maps, self.strides, image_sizes, self.level_embed)
        B, device, dtype = maps[0].shape[0], maps[0].device, maps[0].dtype

        tgt, query_pos, refs = self._learned_queries(B, device, dtype)
        qmask = torch.ones(B, self.cfg.num_queries, dtype=torch.bool, device=device)
        if queries is not None and len(queries) > 0:
            e_tgt, e_pos, e_refs, e_mask = self._external_queries(queries, B, device, dtype)
            tgt = torch.cat([e_tgt, tgt], 1)
            refs = torch.cat([e_refs, refs], 1)
            query_pos = torch.cat([e_pos, query_pos], 1) if query_pos is not None else None
            qmask = torch.cat([e_mask, qmask], 1)
        if self.cfg.query_type == "dab":
            query_pos = self.ref_point_head(sine_embed_2d(refs, self.cfg.d_model).to(dtype))
        ref_sine = sine_embed_2d(refs, self.cfg.d_model).to(dtype) if refs is not None else None

        M = tgt.shape[1]
        eye = torch.eye(M, dtype=torch.bool, device=device)
        self_mask = qmask[:, None, None, :] | eye                       # padded queries attend only to themselves
        cross_mask = flat.mask[:, None, None, :]

        out, layers = tgt, []
        for i, layer in enumerate(self.layers):
            query_sine = None
            if ref_sine is not None:
                query_sine = ref_sine if i == 0 else ref_sine * self.query_scale(out)
            out = layer(out, flat.memory, query_pos, flat.pos, self_mask, cross_mask, query_sine)
            layers.append(self.norm(out))

        extra = {keys.LAYER_TOKENS: layers, keys.QUERY_MASK: qmask, keys.IMAGE_SIZES: image_sizes}
        if refs is not None:
            extra[keys.REFERENCE_POINTS] = refs
        return FeatureMaps(patch_tokens=layers[-1], extra=extra)


query_decoder_configs = {
    "detr_decoder": QueryDecoderConfig(query_type="vanilla", num_queries=100),
    "conditional_detr_decoder": QueryDecoderConfig(query_type="conditional", num_queries=300),
    "dab_detr_decoder": QueryDecoderConfig(query_type="dab", num_queries=300),
}


@Decoder.register(config=query_decoder_configs["detr_decoder"])
def detr_decoder(in_spec: FeatureSpec, cfg: QueryDecoderConfig) -> QueryDecoder:
    return QueryDecoder(in_spec, cfg)


@Decoder.register(config=query_decoder_configs["conditional_detr_decoder"])
def conditional_detr_decoder(in_spec: FeatureSpec, cfg: QueryDecoderConfig) -> QueryDecoder:
    return QueryDecoder(in_spec, cfg)


@Decoder.register(config=query_decoder_configs["dab_detr_decoder"])
def dab_detr_decoder(in_spec: FeatureSpec, cfg: QueryDecoderConfig) -> QueryDecoder:
    return QueryDecoder(in_spec, cfg)
