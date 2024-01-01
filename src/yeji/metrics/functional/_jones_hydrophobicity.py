from typing import Sequence, Union

from torch import Tensor

from yeji.constants import JONES_HYDROPHOBICITY_INDEX

from ._protein_property import _protein_property


def jones_hydrophobicity(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        JONES_HYDROPHOBICITY_INDEX,
    )
