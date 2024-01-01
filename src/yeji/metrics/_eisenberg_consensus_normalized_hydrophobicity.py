from yeji.constants import (
    EISENBERG_CONSENSUS_NORMALIZED_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _ProteinProperty


class EisenbergConsensusNormalizedHydrophobicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            EISENBERG_CONSENSUS_NORMALIZED_HYDROPHOBICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
