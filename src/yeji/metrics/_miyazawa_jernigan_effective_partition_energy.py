from yeji.constants import (
    MIYAZAWA_JERNIGAN_EFFECTIVE_PARTITION_ENERGY_INDEX,
)

from ._protein_property import _ProteinProperty


class MiyazawaJerniganEffectivePartitionEnergy(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            MIYAZAWA_JERNIGAN_EFFECTIVE_PARTITION_ENERGY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
