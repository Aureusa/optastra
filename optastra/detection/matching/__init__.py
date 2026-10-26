from . import iou_matcher as _iou_matcher
from .iou_matcher import *
from . import hungarian as _hungarian
from .hungarian import *

# Module aliases: the star imports rebind names like `iou_matcher` to the registered factory functions.
__all__ = [*_iou_matcher.__all__, *_hungarian.__all__]
