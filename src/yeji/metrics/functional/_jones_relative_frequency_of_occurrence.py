from typing import Sequence, Union

from torch import Tensor

from yeji.constants import JONES_RELATIVE_FREQUENCY_OF_OCCURRENCE_INDEX

from ._protein_property import _protein_property


def jones_relative_frequency_of_occurrence(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        JONES_RELATIVE_FREQUENCY_OF_OCCURRENCE_INDEX,
    )
