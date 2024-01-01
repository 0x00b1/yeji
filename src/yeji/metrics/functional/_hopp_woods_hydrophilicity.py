from typing import Sequence, Union

from torch import Tensor

from yeji.constants import HOPP_WOODS_HYDROPHILICITY_INDEX

from ._protein_property import _protein_property


def hopp_woods_hydrophilicity(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        HOPP_WOODS_HYDROPHILICITY_INDEX,
    )
