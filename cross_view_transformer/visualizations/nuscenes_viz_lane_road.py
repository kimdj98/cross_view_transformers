from .common_lane_road import BaseViz
from ..data.nuscenes_dataset import CLASSES


class NuScenesViz(BaseViz):
    SEMANTICS = CLASSES
