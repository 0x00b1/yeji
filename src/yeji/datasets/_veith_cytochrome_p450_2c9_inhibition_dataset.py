from pathlib import Path

from .__tdc_dataset import _TDCDataset


class VeithCytochromeP4502C9InhibitionDataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4259577,
            suffix="tsv",
            checksum="md5:87d21d2666e8e2bfc76f7d693e060c0c",
            x_columns=["Drug"],
            y_columns=["Y"],
        )
