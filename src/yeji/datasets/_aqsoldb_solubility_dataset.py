from pathlib import Path
from typing import Callable

from ..transforms import Transform
from .__tdc_dataset import _TDCDataset


class AqSolDBSolubilityDataset(_TDCDataset):
    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        *,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
    ):
        r"""

        Parameters
        ----------
        root : str | Path

        download: bool

        transform_fn : Callable | Transform | None

        target_transform_fn : Callable | Transform | None
        """
        super().__init__(
            root=root,
            download=download,
            identifier="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OVHAW8/RLVXZU",
            name="curated-solubility-dataset.tab",
            checksum="5370aa67615adb2f11806ed1aaed37c2bf91e634d36ebaf40509c16d5cede8a0",
            x_columns=["SMILES", "Solubility"],
        )
