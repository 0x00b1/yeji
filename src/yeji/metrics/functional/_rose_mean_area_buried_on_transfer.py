from typing import Sequence, Union

from torch import Tensor

from yeji.constants import ROSE_MEAN_AREA_BURIED_ON_TRANSFER_INDEX

from ._protein_property import _protein_property


def rose_mean_area_buried_on_transfer(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        ROSE_MEAN_AREA_BURIED_ON_TRANSFER_INDEX,
    )
