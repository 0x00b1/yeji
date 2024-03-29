from typing import Sequence

from torch import Tensor

from yeji.constants import FASMAN_OPTICAL_ROTATION_INDEX

from ._protein_property import _protein_property


def fasman_optical_rotation(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        FASMAN_OPTICAL_ROTATION_INDEX,
    )
