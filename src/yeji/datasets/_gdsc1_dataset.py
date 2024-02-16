from pathlib import Path

from .__tdc_dataset import _TDCDataset


class GDSC1Dataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4165726,
            name="gdsc1.pkl",
            checksum="md5:6bee1e2507090559b34ab626e229c0be",
            x_columns=["X1", "X2"],
            y_columns=["Y"],
        )
