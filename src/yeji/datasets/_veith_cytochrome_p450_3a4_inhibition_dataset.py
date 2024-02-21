from pathlib import Path
from typing import Callable

from yeji.transforms import Transform

from .__tdc_dataset import _TDCDataset


class VeithCytochromeP4503A4InhibitionDataset(_TDCDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
    ):
        super().__init__(
            root=root,
            download=download,
            identifier=4259582,
            suffix="tsv",
            checksum="md5:73258e31495abd95072a6e06acbee83a",
            x_columns=["Drug"],
            y_columns=["Y"],
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )
