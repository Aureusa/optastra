from . import rcnn_postprocessor as _rcnn_postprocessor
from .rcnn_postprocessor import *
from . import set_prediction as _set_prediction
from .set_prediction import *

# Module aliases: the star imports rebind names like `iou_matcher` to the registered factory functions.
__all__ = [*_rcnn_postprocessor.__all__, *_set_prediction.__all__]
