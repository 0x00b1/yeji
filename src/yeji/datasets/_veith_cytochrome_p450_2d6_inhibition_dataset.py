from pathlib import Path

from .__tdc_dataset import _TDCDataset


class VeithCytochromeP4502D6InhibitionDataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4259580,
            suffix="tsv",
            checksum="md5:9f82eae1ecccec93c8fc4249955e8694",
            x_columns=["Drug"],
            y_columns=["Y"],
        )