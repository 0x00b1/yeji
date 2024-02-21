from typing import Sequence

from torch import Tensor

from yeji.constants import HOPP_WOODS_HYDROPHILICITY_INDEX

from ._protein_property import _protein_property


def hopp_woods_hydrophilicity(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        HOPP_WOODS_HYDROPHILICITY_INDEX,
    )
