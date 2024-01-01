from yeji.constants import ROSE_MEAN_FRACTIONAL_AREA_LOSS_INDEX

from ._protein_property import _ProteinProperty


class RoseMeanFractionalAreaLoss(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            ROSE_MEAN_FRACTIONAL_AREA_LOSS_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
