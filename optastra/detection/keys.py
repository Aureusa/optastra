"""String keys the R-CNN family passes through ``FeatureMaps.extra`` / ``HeadOutput.extra``.

``HeadOutput`` has no typed slot for multi-stage outputs yet, so architectures,
criteria and postprocessors talk to each other through these ``extra`` keys.
This module is the single place they are spelled out -- import the constants
instead of writing the strings, so a later typed-output redesign only has to
follow the references to this file.

Detector features (``FeatureMaps.extra`` handed to the proposal generator):
    IMAGE_SIZES            list[tuple[int, int]]  per-image (h, w) before batch padding

Proposal generator output (``FeatureMaps.extra`` of the RPN):
    ANCHORS                (A, 4) float          anchors of every level, concatenated (XYXY)
    OBJECTNESS_LOGITS      (N, A)                one logit per anchor, same order as ANCHORS
    BBOX_DELTAS            (N, A, 4)             one delta per anchor, same order as ANCHORS
    NUM_ANCHORS_PER_LEVEL  list[int]             how many of the A anchors belong to each level
    BOX_CODER_WEIGHTS      tuple of 4 floats     weights the RPN decodes deltas with (the loss
                                                 must encode its targets with the same weights)
    IMAGE_SIZES            list[tuple[int, int]] copied from the detector features
    PROPOSALS              (P, 5) float          detached (batch_index, x1, y1, x2, y2)
    PROPOSAL_SCORES        (P,)                  detached objectness logits of PROPOSALS

Architecture output (``HeadOutput.extra`` of Fast / Faster / Mask R-CNN):
    ROI_BOXES              (R, 5) float          (batch_index, x1, y1, x2, y2) boxes the ROI
                                                 heads ran on; row i matches logits[i]
    RPN                    FeatureMaps           the proposal generator output (Faster/Mask only)
    IMAGE_SIZES            list[tuple[int, int]] one entry per image -- also defines batch size

Set-prediction outputs (DETR family; ``HeadOutput`` of ``detr_box_head`` / ``keypoint_head``
and of the ``DETR`` architecture):
    logits / boxes         (B, M, K+1) / (B, M, 4)  class logits (last = no object) and boxes as
                                                 normalised (cx, cy, w, h) -- fractions of each
                                                 image's size -- in the typed HeadOutput fields
    scores                 (B, M)                objectness logits (keypoint_head)
    POINTS                 (B, M, 2) float       points (x, y) as fractions of each image's size
    POINT_LOG_SCALES       (B, M, 2) float       optional log of the predicted point scale, same units
    AUX_OUTPUTS            list[HeadOutput]      the head applied to every earlier decoder layer
    QUERY_MASK             (B, M) bool           real queries (``optastra.decoders.keys.QUERY_MASK``)
    IMAGE_SIZES            list[tuple[int, int]] one entry per image
    DECODER_TOKENS         (B, M, D)             the last decoder layer's tokens (DETR architecture)
"""
from __future__ import annotations

__all__ = [
    "IMAGE_SIZES",
    "ANCHORS",
    "OBJECTNESS_LOGITS",
    "BBOX_DELTAS",
    "NUM_ANCHORS_PER_LEVEL",
    "BOX_CODER_WEIGHTS",
    "PROPOSALS",
    "PROPOSAL_SCORES",
    "ROI_BOXES",
    "RPN",
    "POINTS",
    "POINT_LOG_SCALES",
    "AUX_OUTPUTS",
    "QUERY_MASK",
    "DECODER_TOKENS",
]

IMAGE_SIZES = "image_sizes"

ANCHORS = "anchors"
OBJECTNESS_LOGITS = "objectness_logits"
BBOX_DELTAS = "bbox_deltas"
NUM_ANCHORS_PER_LEVEL = "num_anchors_per_level"
BOX_CODER_WEIGHTS = "box_coder_weights"
PROPOSALS = "proposals"
PROPOSAL_SCORES = "proposal_scores"

ROI_BOXES = "roi_boxes"
RPN = "rpn"

POINTS = "points"
POINT_LOG_SCALES = "point_log_scales"
AUX_OUTPUTS = "aux_outputs"
QUERY_MASK = "query_mask"   # same string as optastra.decoders.keys.QUERY_MASK
DECODER_TOKENS = "decoder_tokens"
