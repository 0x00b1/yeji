from yeji.constants import (
    BLACK_MOULD_SCALED_SIDE_CHAIN_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _ProteinProperty


class BlackMouldScaledSideChainHydrophobicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            BLACK_MOULD_SCALED_SIDE_CHAIN_HYDROPHOBICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
