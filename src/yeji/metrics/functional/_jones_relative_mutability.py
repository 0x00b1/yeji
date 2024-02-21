from typing import Sequence

from torch import Tensor

from yeji.constants import JONES_RELATIVE_MUTABILITY_INDEX

from ._protein_property import _protein_property


def jones_relative_mutability(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        JONES_RELATIVE_MUTABILITY_INDEX,
    )
