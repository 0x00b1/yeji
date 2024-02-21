from typing import Sequence

from torch import Tensor

from yeji.constants import JONES_HYDROPHOBICITY_INDEX

from ._protein_property import _protein_property


def jones_hydrophobicity(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        JONES_HYDROPHOBICITY_INDEX,
    )
