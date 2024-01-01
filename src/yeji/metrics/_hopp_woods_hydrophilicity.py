from yeji.constants import HOPP_WOODS_HYDROPHILICITY_INDEX

from ._protein_property import _ProteinProperty


class HoppWoodsHydrophilicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            HOPP_WOODS_HYDROPHILICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
