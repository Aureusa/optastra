from . import rcnn_postprocessor as _rcnn_postprocessor
from .rcnn_postprocessor import *

# Module aliases: the star imports rebind names like `iou_matcher` to the registered factory functions.
__all__ = [*_rcnn_postprocessor.__all__]
