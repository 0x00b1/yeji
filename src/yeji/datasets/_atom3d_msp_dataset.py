from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from pandas import DataFrame
from torch import Tensor

from yeji.transforms import Transform

from ._atom3d_dataset import ATOM3DDataset


class ATOM3DMSPDataset(ATOM3DDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
    ):
        r"""

        Parameters
        ----------
        root : str | Path
            Root directory of dataset.

        download: bool
            If `True`, downloads the dataset to the root directory. If dataset
            already exists, it is not redownloaded. Default, `False`.

        transform_fn : Callable | Transform | None
            Transforms the input.

        target_transform_fn : Callable | Transform | None
            Transforms the target.
        """
        super().__init__(
            root,
            "raw/MSP/data",
            "https://zenodo.org/record/4962515/files/MSP-raw.tar.gz",
            "MSP",
            checksum="77aeb79cfc80bd51cdfb2aa321bf6128",
            download=download,
        )

        self._transform_fn = transform_fn

        self._target_transform_fn = target_transform_fn

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Tuple[DataFrame, DataFrame], Tensor]:
        item = super().__getitem__(index)

        structure = DataFrame(**item["original_atoms"])

        mutant = DataFrame(**item["mutated_atoms"])

        if self._transform_fn is not None:
            structure, mutant = self._transform_fn(structure, mutant)

        target = torch.tensor(int(item["label"]))

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return (structure, mutant), target
