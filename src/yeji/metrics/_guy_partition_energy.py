from yeji.constants import GUY_PARTITION_ENERGY_INDEX

from ._protein_property import _ProteinProperty


class GuyPartitionEnergy(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            GUY_PARTITION_ENERGY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
