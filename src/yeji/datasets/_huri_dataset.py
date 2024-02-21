from pathlib import Path
from typing import Callable

from yeji.transforms import Transform

from .__tdc_dataset import _TDCDataset


class HuRIDataset(_TDCDataset):
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
            identifier=4139567,
            suffix="huri.tab",
            checksum="md5:d934f40f048fc8686c0137c273ceec57",
            x_columns=["X1", "X2"],
            y_columns=["Y"],
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )
