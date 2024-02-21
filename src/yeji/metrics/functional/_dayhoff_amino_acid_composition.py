from typing import Sequence

from torch import Tensor

from yeji.constants import DAYHOFF_AMINO_ACID_COMPOSITION_INDEX

from ._protein_property import _protein_property


def dayhoff_amino_acid_composition(
    predictions: str | Sequence[str],
) -> Tensor:
    return _protein_property(
        predictions,
        DAYHOFF_AMINO_ACID_COMPOSITION_INDEX,
    )
