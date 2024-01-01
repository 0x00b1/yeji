from yeji.constants import GOLDSACK_CHALIFOUX_HYDROPHOBICITY_FACTOR_INDEX

from ._protein_property import _ProteinProperty


class GoldsackChalifouxHydrophobicityFactor(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            GOLDSACK_CHALIFOUX_HYDROPHOBICITY_FACTOR_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
