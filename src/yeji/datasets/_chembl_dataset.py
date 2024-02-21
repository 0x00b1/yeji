from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from yeji.transforms import Transform


class ChEMBLDataset(Dataset):
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
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
