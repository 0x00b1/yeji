from pathlib import Path

from .__tdc_dataset import _TDCDataset


class DisGeNETDataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4168282,
            name="disgenet.csv",
            checksum="md5:b7efdf1dc006ff04a33bb3a4aec5d746",
            x_columns=["X1", "ID2"],
            y_columns=["Y"],
            sep=",",
        )
