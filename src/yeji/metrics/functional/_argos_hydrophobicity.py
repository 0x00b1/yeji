from typing import Sequence

from torch import Tensor

from yeji.constants import ARGOS_HYDROPHOBICITY_INDEX

from ._protein_property import _protein_property


def argos_hydrophobicity(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        ARGOS_HYDROPHOBICITY_INDEX,
    )
