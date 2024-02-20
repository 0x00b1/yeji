from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from yeji.features import EulerAngle
from yeji.transforms import ToEulerAngle, Transform


class RandomEulerAngleDataset(Dataset):
    def __init__(
        self,
        size: int = 2048,
        axes: str = "xyz",
        degrees: bool = False,
        transform: Callable | Transform | None = None,
    ):
        super().__init__()

        self._size = size

        self._axes = axes

        self._degrees = degrees

        self._transform = transform

    def __getitem__(self, index: int) -> EulerAngle | Any:
        if index >= len(self):
            raise IndexError

        item = torch.randn(3)

        item = ToEulerAngle()(item, axes=self._axes, degrees=self._degrees)

        if self._transform is not None:
            item = self._transform(item)

        return item

    def __len__(self) -> int:
        return self._size
