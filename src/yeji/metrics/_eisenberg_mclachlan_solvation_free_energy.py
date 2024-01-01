from yeji.constants import EISENBERG_MCLACHLAN_SOLVATION_FREE_ENERGY_INDEX

from ._protein_property import _ProteinProperty


class EisenbergMclachlanSolvationFreeEnergy(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            EISENBERG_MCLACHLAN_SOLVATION_FREE_ENERGY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
