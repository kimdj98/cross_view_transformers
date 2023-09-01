from .common_vel import BaseViz
from ..data.nuscenes_dataset import CLASSES


class NuScenesViz(BaseViz):
    SEMANTICS = CLASSES
