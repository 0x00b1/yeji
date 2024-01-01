from yeji.constants import JUKES_AMINO_ACID_DISTRIBUTION_INDEX

from ._protein_property import _ProteinProperty


class JukesAminoAcidDistribution(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            JUKES_AMINO_ACID_DISTRIBUTION_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
