from .common_wpp import BaseViz
from ..data.nuscenes_dataset import CLASSES


class NuScenesViz(BaseViz):
    SEMANTICS = CLASSES
