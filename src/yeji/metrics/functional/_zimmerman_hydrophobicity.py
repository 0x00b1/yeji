from typing import Sequence

from torch import Tensor

from yeji.constants import ZIMMERMAN_HYDROPHOBICITY_INDEX

from ._protein_property import _protein_property


def zimmerman_hydrophobicity(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        ZIMMERMAN_HYDROPHOBICITY_INDEX,
    )
