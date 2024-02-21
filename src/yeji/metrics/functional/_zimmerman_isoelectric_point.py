from typing import Sequence

from torch import Tensor

from yeji.constants import ZIMMERMAN_ISOELECTRIC_POINT_INDEX

from ._protein_property import _protein_property


def zimmerman_isoelectric_point(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        ZIMMERMAN_ISOELECTRIC_POINT_INDEX,
    )
