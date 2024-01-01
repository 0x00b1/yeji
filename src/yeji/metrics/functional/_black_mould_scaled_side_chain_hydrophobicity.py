from typing import Sequence, Union

from torch import Tensor

from yeji.constants import (
    BLACK_MOULD_SCALED_SIDE_CHAIN_HYDROPHOBICITY_INDEX,
)

from ._protein_property import _protein_property


def black_mould_scaled_side_chain_hydrophobicity(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        BLACK_MOULD_SCALED_SIDE_CHAIN_HYDROPHOBICITY_INDEX,
    )
