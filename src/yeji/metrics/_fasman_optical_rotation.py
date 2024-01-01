from yeji.constants import FASMAN_OPTICAL_ROTATION_INDEX

from ._protein_property import _ProteinProperty


class FasmanOpticalRotation(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            FASMAN_OPTICAL_ROTATION_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
