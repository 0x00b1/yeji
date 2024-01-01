from typing import Sequence, Union

from torch import Tensor

from yeji.constants import (
    EISENBERG_MCLACHLAN_SOLVATION_FREE_ENERGY_INDEX,
)

from ._protein_property import _protein_property


def eisenberg_mclachlan_solvation_free_energy(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        EISENBERG_MCLACHLAN_SOLVATION_FREE_ENERGY_INDEX,
    )
