"""String keys decoders pass through ``FeatureMaps.extra``.

Decoder output (``FeatureMaps`` with ``patch_tokens`` = the last layer's tokens):
    LAYER_TOKENS       list[(B, M, D)]  every layer's (normalised) output, last one included --
                                        heads run on each for auxiliary losses
    REFERENCE_POINTS   (B, M, 2)        the queries' reference points, (x, y) as fractions of
                                        each image's size; absent for vanilla DETR queries
    QUERY_MASK         (B, M) bool      real queries (False for padded external queries)
    IMAGE_SIZES        list[(h, w)]     per-image size before padding (copied from the input;
                                        same string as ``optastra.detection.keys.IMAGE_SIZES``)
"""
from __future__ import annotations

__all__ = ["LAYER_TOKENS", "REFERENCE_POINTS", "QUERY_MASK", "IMAGE_SIZES"]

LAYER_TOKENS = "layer_tokens"
REFERENCE_POINTS = "reference_points"
QUERY_MASK = "query_mask"
IMAGE_SIZES = "image_sizes"
