from typing import Sequence

from torch import Tensor

from yeji.constants import GOLDSACK_CHALIFOUX_HYDROPHOBICITY_FACTOR_INDEX

from ._protein_property import _protein_property


def goldsack_chalifoux_hydrophobicity_factor(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        GOLDSACK_CHALIFOUX_HYDROPHOBICITY_FACTOR_INDEX,
    )
