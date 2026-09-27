from . import iou_matcher as _iou_matcher
from .iou_matcher import *

# Module aliases: the star imports rebind names like `iou_matcher` to the registered factory functions.
__all__ = [*_iou_matcher.__all__]
