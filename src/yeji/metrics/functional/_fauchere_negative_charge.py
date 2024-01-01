from typing import Sequence, Union

from torch import Tensor

from yeji.constants import FAUCHERE_NEGATIVE_CHARGE_INDEX

from ._protein_property import _protein_property


def fauchere_negative_charge(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        FAUCHERE_NEGATIVE_CHARGE_INDEX,
    )
