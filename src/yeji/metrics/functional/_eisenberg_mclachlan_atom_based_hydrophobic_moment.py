from typing import Sequence

from torch import Tensor

from yeji.constants import (
    EISENBERG_MCLACHLAN_ATOM_BASED_HYDROPHOBIC_MOMENT_INDEX,
)

from ._protein_property import _protein_property


def eisenberg_mclachlan_atom_based_hydrophobic_moment(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        EISENBERG_MCLACHLAN_ATOM_BASED_HYDROPHOBIC_MOMENT_INDEX,
    )
