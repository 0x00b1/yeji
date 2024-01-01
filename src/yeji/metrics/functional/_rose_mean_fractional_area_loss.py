from typing import Sequence, Union

from torch import Tensor

from yeji.constants import ROSE_MEAN_FRACTIONAL_AREA_LOSS_INDEX

from ._protein_property import _protein_property


def rose_mean_fractional_area_loss(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        ROSE_MEAN_FRACTIONAL_AREA_LOSS_INDEX,
    )
