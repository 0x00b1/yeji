from yeji.constants import WOLFENDEN_HYDROPHOBICITY_INDEX

from ._protein_property import _ProteinProperty


class WolfendenHydrophobicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            WOLFENDEN_HYDROPHOBICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
