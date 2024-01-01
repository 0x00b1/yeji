from typing import Sequence, Union

from torch import Tensor

from yeji.constants import ZAMYATNIN_PROTEIN_VOLUME_IN_SOLUTION_INDEX

from ._protein_property import _protein_property


def zamyatnin_protein_volume_in_solution(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        ZAMYATNIN_PROTEIN_VOLUME_IN_SOLUTION_INDEX,
    )
