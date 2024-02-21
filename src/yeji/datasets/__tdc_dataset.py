from pathlib import Path
from typing import Callable

import pandas
import pooch
from torch.utils.data import Dataset

from yeji.transforms import Transform


class _TDCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        *,
        identifier: int,
        suffix: str,
        checksum: str,
        x_columns: list[str],
        y_columns: list[str],
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
    ):
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        if download:
            pooch.retrieve(
                f"https://dataverse.harvard.edu/api/access/datafile/{identifier}",
                fname=f"{self.__class__.__name__}.{suffix}",
                known_hash=checksum,
                path=root / self.__class__.__name__,
            )

        path = (
            root
            / self.__class__.__name__
            / f"{self.__class__.__name__}.{suffix}"
        )

        match path.suffix:
            case ".csv":
                self._data = pandas.read_csv(path)
            case ".pkl":
                self._data = pandas.read_pickle(path)
            case ".tab" | ".tsv":
                self._data = pandas.read_csv(path, sep="\t")
            case _:
                raise ValueError

        self._transform_fn = transform_fn
        self._target_transform_fn = target_transform_fn

        self._xs = self._data[x_columns].apply(tuple, axis=1)
        self._ys = self._data[y_columns].apply(tuple, axis=1)

    def __getitem__(self, index: int):
        x = self._xs[index]

        if len(x) == 1:
            x = x[0]

        if self._transform_fn is not None:
            x = self._transform_fn(x)

        y = self._ys[index]

        if len(y) == 1:
            y = y[0]

        if self._target_transform_fn is not None:
            y = self._target_transform_fn(y)

        return x, y

    def __len__(self) -> int:
        return len(self._data)
