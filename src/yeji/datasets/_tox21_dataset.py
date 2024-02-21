from pathlib import Path
from typing import Any, Callable

from torch.utils.data import Dataset

from yeji.transforms import Transform


class Tox21Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
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
        super().__init__()

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
