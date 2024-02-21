from typing import Sequence

from torch import Tensor

from yeji.constants import (
    JURETIC_MODIFIED_KYTE_DOOLITTLE_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _protein_property


def juretic_modified_kyte_doolittle_hydrophobicity(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        JURETIC_MODIFIED_KYTE_DOOLITTLE_HYDROPHOBICITY_INDEX,
    )
