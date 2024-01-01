from yeji.constants import KYTE_DOOLITTLE_HYDROPATHY_INDEX

from ._protein_property import _ProteinProperty


class KyteDoolittleHydropathy(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            KYTE_DOOLITTLE_HYDROPATHY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
