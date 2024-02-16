from pathlib import Path

from .__tdc_dataset import _TDCDataset


class VeithCytochromeP4502C19InhibitionDataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4259576,
            suffix="tsv",
            checksum="md5:fe0c4420effb5df2417fa9c9a2ba07ae",
            x_columns=["Drug"],
            y_columns=["Y"],
        )
