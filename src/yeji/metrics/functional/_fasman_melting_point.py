from typing import Sequence, Union

from torch import Tensor

from yeji.constants import FASMAN_MELTING_POINT_INDEX

from ._protein_property import _protein_property


def fasman_melting_point(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        FASMAN_MELTING_POINT_INDEX,
    )
