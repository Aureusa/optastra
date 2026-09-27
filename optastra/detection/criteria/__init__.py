from . import detr_stub as _detr_stub
from .detr_stub import *
from . import rcnn as _rcnn
from .rcnn import *

# Module aliases: the star imports rebind names like `iou_matcher` to the registered factory functions.
__all__ = [*_detr_stub.__all__, *_rcnn.__all__]
