from typing import Sequence, Union

from torch import Tensor

from yeji.constants import FAUCHERE_POSITIVE_CHARGE_INDEX

from ._protein_property import _protein_property


def fauchere_positive_charge(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        FAUCHERE_POSITIVE_CHARGE_INDEX,
    )
