from typing import Sequence, Union

from torch import Tensor

from yeji.constants import (
    MIYAZAWA_JERNIGAN_EFFECTIVE_PARTITION_ENERGY_INDEX,
)

from ._protein_property import _protein_property


def miyazawa_jernigan_effective_partition_energy(
    predictions: Union[str, Sequence[str]],
) -> Tensor:
    return _protein_property(
        predictions,
        MIYAZAWA_JERNIGAN_EFFECTIVE_PARTITION_ENERGY_INDEX,
    )
