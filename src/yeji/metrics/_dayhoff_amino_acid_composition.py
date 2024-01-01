from yeji.constants import DAYHOFF_AMINO_ACID_COMPOSITION_INDEX

from ._protein_property import _ProteinProperty


class DayhoffAminoAcidComposition(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            DAYHOFF_AMINO_ACID_COMPOSITION_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
