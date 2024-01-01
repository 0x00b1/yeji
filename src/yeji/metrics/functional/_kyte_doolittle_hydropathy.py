from typing import Sequence, Union

from torch import Tensor

from yeji.constants import KYTE_DOOLITTLE_HYDROPATHY_INDEX

from ._protein_property import _protein_property


def kyte_doolittle_hydropathy(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        KYTE_DOOLITTLE_HYDROPATHY_INDEX,
    )
