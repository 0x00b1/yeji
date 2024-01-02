from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import Dataset

from yeji.features import EulerAngles
from yeji.transforms import ToEulerAngles, Transform


class RandomEulerAnglesDataset(Dataset):
    def __init__(
        self,
        size: int = 2048,
        axes: str = "xyz",
        degrees: bool = False,
        transform: Optional[Union[Callable, Transform]] = None,
    ):
        super().__init__()

        self._size = size

        self._axes = axes

        self._degrees = degrees

        self._transform = transform

    def __getitem__(self, index: int) -> Union[EulerAngles, Any]:
        if index >= len(self):
            raise IndexError

        item = torch.randn(3)

        item = ToEulerAngles()(item)

        if self._transform is not None:
            item = self._transform(item)

        return item

    def __len__(self) -> int:
        return self._size
