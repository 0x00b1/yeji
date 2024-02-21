from pathlib import Path
from typing import Callable

from yeji.transforms import Transform

from .__tdc_dataset import _TDCDataset


class VeithCytochromeP4502C19InhibitionDataset(_TDCDataset):
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
            identifier=4259576,
            suffix="tsv",
            checksum="md5:fe0c4420effb5df2417fa9c9a2ba07ae",
            x_columns=["Drug"],
            y_columns=["Y"],
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )
