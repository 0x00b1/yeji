from pathlib import Path
from typing import Callable

from ..transforms import Transform
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
