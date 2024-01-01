from typing import Sequence, Union

from torch import Tensor

from yeji.constants import (
    EISENBERG_MCLACHLAN_DIRECTION_OF_HYDROPHOBIC_MOMENT_INDEX,
)

from ._protein_property import _protein_property


def eisenberg_mclachlan_direction_of_hydrophobic_moment(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        EISENBERG_MCLACHLAN_DIRECTION_OF_HYDROPHOBIC_MOMENT_INDEX,
    )
