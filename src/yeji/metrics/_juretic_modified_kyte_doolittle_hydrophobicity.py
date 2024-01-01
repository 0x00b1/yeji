from yeji.constants import (
    JURETIC_MODIFIED_KYTE_DOOLITTLE_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _ProteinProperty


class JureticModifiedKyteDoolittleHydrophobicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            JURETIC_MODIFIED_KYTE_DOOLITTLE_HYDROPHOBICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
