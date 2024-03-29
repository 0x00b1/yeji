from pathlib import Path
from typing import Callable

from yeji.transforms import Transform

from .__tdc_dataset import _TDCDataset


class AqSolDBSolubilityDataset(_TDCDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
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
            root=root,
            download=download,
            identifier=0,
            suffix="curated-solubility-dataset.tab",
            checksum="5370aa67615adb2f11806ed1aaed37c2bf91e634d36ebaf40509c16d5cede8a0",
            x_columns=["SMILES"],
            y_columns=["Solubility"],
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )
