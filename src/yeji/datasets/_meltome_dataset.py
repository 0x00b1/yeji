from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import pandas
import torch
from prescient.transforms import Transform
from torch import Tensor

from ._data_frame_dataset import DataFrameDataset


class MeltomeDataset(DataFrameDataset):
    """
    The meltome contains "melting behaviour" and a "melting degree"
    for proteins in human and other species. From "Meltome atlas—thermal
    proteome stability across the tree of life".
    Accessed at https://github.com/J-SNACKKB/FLIP/tree/main/splits/meltome
    """

    def __init__(
        self,
        root: Union[str, Path],
        path: Union[
            str, Path
        ] = "s3://prescient-data-dev/sandbox/freyn6/flip/meltome.csv.zip",
        *,
        columns: Optional[Sequence[str]] = None,
        target_columns: Optional[Sequence[str]] = None,
        train: bool = True,
        split: str = "human",
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
        **kwargs,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param columns: x features of the dataset. items in the dataset are
            of the form ((columns), (target_columns)).

        :param target_columns: y features of the dataset. items in the dataset
            are of the form ((columns), (target_columns)).

        :param train: If ``True``, use the training set, otherwise, return the
            test set (default: ``True``).

        :param split: one of "human", "human_cell", "mixed_split"
            that specifies train/test split from FLIP

        :param transform_fn: A ``Callable`` or ``Transform`` that maps data to
            transformed data (default: ``None``).

        :param target_transform_fn: ``Callable`` or ``Transform`` that maps a
            target to a transformed target (default: ``None``).
        """
        super().__init__(
            root,
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )

        self._path = path
        allowed_splits = ["human", "human_cell", "mixed_split"]
        if split not in allowed_splits:
            raise ValueError(f"split not one of {allowed_splits}")

        self._split = split

        if columns is not None:
            self._columns = columns
        else:
            self._columns = ["sequence", self._split]

        if target_columns is not None:
            self._target_columns = target_columns
        else:
            self._target_columns = ["target"]

        self._train = train

        subset = self._columns

        if train:
            subset = [*self._columns, *self._target_columns]

        self._data = pandas.read_csv(self._path, **kwargs)

        self._data = self._data.dropna(subset=subset)

        if train:
            self._data = self._data[self._data[self._split] == "train"]
        else:
            self._data = self._data[self._data[self._split] == "test"]

        self._data = self._data.drop(axis=1, columns=self._split)

    def __getitem__(
        self, index: int
    ) -> Tuple[Tuple[str, ...], Tuple[Tensor, ...]]:
        item = super().__getitem__(index)

        x = tuple(item[col] for col in self._columns if col != self._split)

        if self._transform_fn is not None:
            x = self._transform_fn(x)

        y = tuple(item[col] for col in self._target_columns)

        if self._target_transform_fn is not None:
            y = self._target_transform_fn(y)

        if not all(isinstance(y_val, Tensor) for y_val in y):
            y = tuple(torch.tensor(y_val) for y_val in y)

        return x, y
