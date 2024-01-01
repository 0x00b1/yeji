from typing import Sequence, Union

from torch import Tensor

from yeji.constants import WOLFENDEN_HYDROPHOBICITY_INDEX

from ._protein_property import _protein_property


def wolfenden_hydrophobicity(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        WOLFENDEN_HYDROPHOBICITY_INDEX,
    )
