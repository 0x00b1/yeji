from yeji.constants import CID_NORMALIZED_AVERAGE_HYDROPHOBICITY_INDEX

from ._protein_property import _ProteinProperty


class CidNormalizedAverageHydrophobicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            CID_NORMALIZED_AVERAGE_HYDROPHOBICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
