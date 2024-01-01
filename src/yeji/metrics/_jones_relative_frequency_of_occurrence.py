from yeji.constants import JONES_RELATIVE_FREQUENCY_OF_OCCURRENCE_INDEX

from ._protein_property import _ProteinProperty


class JonesRelativeFrequencyOfOccurrence(_ProteinProperty):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(
            JONES_RELATIVE_FREQUENCY_OF_OCCURRENCE_INDEX,
            dist_sync_on_step=dist_sync_on_step,
        )
