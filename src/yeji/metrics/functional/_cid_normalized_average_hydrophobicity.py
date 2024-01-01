from typing import Sequence, Union

from torch import Tensor

from yeji.constants import CID_NORMALIZED_AVERAGE_HYDROPHOBICITY_INDEX

from ._protein_property import _protein_property


def cid_normalized_average_hydrophobicity(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        CID_NORMALIZED_AVERAGE_HYDROPHOBICITY_INDEX,
    )
