from typing import Sequence, Union

from torch import Tensor

from yeji.constants import ZIMMERMAN_BULKINESS_INDEX

from ._protein_property import _protein_property


def zimmerman_bulkiness(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        ZIMMERMAN_BULKINESS_INDEX,
    )
