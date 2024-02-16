from pathlib import Path

from .__tdc_dataset import _TDCDataset


class GDSC2Dataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4165727,
            name="gdsc2.pkl",
            checksum="md5:217ccb2c49dc43485924f8678eaf7e34",
            x_columns=["X1", "X2"],
            y_columns=["Y"],
        )
