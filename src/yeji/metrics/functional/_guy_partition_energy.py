from typing import Sequence, Union

from torch import Tensor

from yeji.constants import GUY_PARTITION_ENERGY_INDEX

from ._protein_property import _protein_property


def guy_partition_energy(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        GUY_PARTITION_ENERGY_INDEX,
    )
