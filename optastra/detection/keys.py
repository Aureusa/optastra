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
