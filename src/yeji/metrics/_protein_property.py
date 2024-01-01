from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from torch import Tensor
from torchmetrics import Metric

from .functional._protein_property import (
    _protein_property_compute,
    _protein_property_update,
)


class _ProteinProperty(Metric):
    full_state_update: bool = False
    higher_is_better: bool = False
    is_differentiable: bool = False
    plot_lower_bound: float = 0.0

    score: Tensor
    count: Tensor

    def __init__(
        self,
        index: Dict[str, float],
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.index = index

        self.add_state(
            "score",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "count",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )

    def compute(self) -> Tensor:
        return _protein_property_compute(self.score, self.count)

    def plot(
        self,
        scores: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        axis: Optional[Axes] = None,
    ) -> Tuple[Figure, Union[Axes, ndarray]]:
        return self._plot(scores, axis)

    def update(self, predictions: Union[str, Sequence[str]]):
        scores = _protein_property_update(predictions, index=self.index)

        self.score = torch.add(self.score, torch.sum(scores))
        self.count = torch.add(self.count, scores.shape[0])
