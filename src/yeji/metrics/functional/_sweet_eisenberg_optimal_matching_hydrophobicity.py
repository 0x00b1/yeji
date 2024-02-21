from typing import Sequence

from torch import Tensor

from yeji.constants import (
    SWEET_EISENBERG_OPTIMAL_MATCHING_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _protein_property


def sweet_eisenberg_optimal_matching_hydrophobicity(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        SWEET_EISENBERG_OPTIMAL_MATCHING_HYDROPHOBICITY_INDEX,
    )
