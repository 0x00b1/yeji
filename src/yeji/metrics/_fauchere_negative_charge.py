from yeji.constants import FAUCHERE_NEGATIVE_CHARGE_INDEX

from ._protein_property import _ProteinProperty


class FauchereNegativeCharge(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            FAUCHERE_NEGATIVE_CHARGE_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
