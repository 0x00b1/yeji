from yeji.constants import (
    EISENBERG_MCLACHLAN_DIRECTION_OF_HYDROPHOBIC_MOMENT_INDEX,
)

from ._protein_property import _ProteinProperty


class EisenbergMclachlanDirectionOfHydrophobicMoment(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            EISENBERG_MCLACHLAN_DIRECTION_OF_HYDROPHOBIC_MOMENT_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
