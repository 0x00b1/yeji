from typing import Sequence, Union

from torch import Tensor

from yeji.constants import FASMAN_MOLECULAR_WEIGHT_INDEX

from ._protein_property import _protein_property


def fasman_molecular_weight(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        FASMAN_MOLECULAR_WEIGHT_INDEX,
    )
