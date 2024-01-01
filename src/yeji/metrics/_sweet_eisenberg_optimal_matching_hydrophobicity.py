from yeji.constants import (
    SWEET_EISENBERG_OPTIMAL_MATCHING_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _ProteinProperty


class SweetEisenbergOptimalMatchingHydrophobicity(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            SWEET_EISENBERG_OPTIMAL_MATCHING_HYDROPHOBICITY_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
