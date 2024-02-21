from typing import Sequence

from torch import Tensor

from yeji.constants import FASMAN_MOLECULAR_WEIGHT_INDEX

from ._protein_property import _protein_property


def fasman_molecular_weight(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        FASMAN_MOLECULAR_WEIGHT_INDEX,
    )
