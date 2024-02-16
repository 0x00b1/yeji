from pathlib import Path

from .__tdc_dataset import _TDCDataset


class HuRIDataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4139567,
            name="huri.tab",
            checksum="md5:d934f40f048fc8686c0137c273ceec57",
            x_columns=["X1", "X2"],
            y_columns=["Y"],
        )
