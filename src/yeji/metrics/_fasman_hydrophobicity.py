from yeji.constants import FASMAN_HYDROPHOBICITY_INDEX

from ._protein_property import _ProteinProperty


class FasmanHydrophobicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            FASMAN_HYDROPHOBICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
