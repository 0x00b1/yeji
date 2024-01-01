from yeji.constants import ZAMYATNIN_PROTEIN_VOLUME_IN_SOLUTION_INDEX

from ._protein_property import _ProteinProperty


class ZamyatninProteinVolumeInSolution(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            ZAMYATNIN_PROTEIN_VOLUME_IN_SOLUTION_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
