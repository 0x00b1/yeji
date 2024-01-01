from yeji.constants import ZIMMERMAN_ISOELECTRIC_POINT_INDEX

from ._protein_property import _ProteinProperty


class ZimmermanIsoelectricPoint(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            ZIMMERMAN_ISOELECTRIC_POINT_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
