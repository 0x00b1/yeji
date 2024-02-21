from pathlib import Path
from typing import Callable

from ..transforms import Transform
from .__tdc_dataset import _TDCDataset


class GDSC2Dataset(_TDCDataset):
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
            identifier=4165727,
            suffix="gdsc2.pkl",
            checksum="md5:217ccb2c49dc43485924f8678eaf7e34",
            x_columns=["X1", "X2"],
            y_columns=["Y"],
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )
