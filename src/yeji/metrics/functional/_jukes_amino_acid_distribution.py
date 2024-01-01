from typing import Sequence, Union

from torch import Tensor

from yeji.constants import JUKES_AMINO_ACID_DISTRIBUTION_INDEX

from ._protein_property import _protein_property


def jukes_amino_acid_distribution(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        JUKES_AMINO_ACID_DISTRIBUTION_INDEX,
    )
