from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import torch
from pandas import DataFrame
from torch import Tensor

from yeji.transforms import Transform

from ._atom3d_dataset import ATOM3DDataset


class ATOM3DPSRDataset(ATOM3DDataset):
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
            "raw/casp5_to_13/data",
            "https://zenodo.org/record/4915648/files/PSR-raw.tar.gz",
            "PSR",
            checksum="80caef3c98febb70951fa244c8303039",
            download=download,
        )

        self._transform_fn = transform_fn

        self._target_transform_fn = target_transform_fn

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[DataFrame, Dict[str, Tensor]]:
        item = super().__getitem__(index)

        features = DataFrame(**item["atoms"])

        if self._transform_fn is not None:
            features = self._transform_fn(features)

        target = item["scores"]

        for k, v in target.items():
            target[k] = torch.tensor(target[k])

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return features, target
