from typing import Sequence

from torch import Tensor

from yeji.constants import (
    EISENBERG_CONSENSUS_NORMALIZED_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _protein_property


def eisenberg_consensus_normalized_hydrophobicity(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        EISENBERG_CONSENSUS_NORMALIZED_HYDROPHOBICITY_INDEX,
    )
