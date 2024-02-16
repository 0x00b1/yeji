from pathlib import Path

from .__tdc_dataset import _TDCDataset


class VeithCytochromeP4501A2InhibitionDataset(_TDCDataset):
    def __init__(self, root: str | Path, download: bool = False):
        super().__init__(
            root=root,
            download=download,
            identifier=4259573,
            suffix="tsv",
            checksum="md5:e5eeb84ca332cd059c73b816f7964193",
            x_columns=["Drug"],
            y_columns=["Y"],
        )
