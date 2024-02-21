from pathlib import Path
from typing import Any, Callable

from yeji.transforms import Transform

from ._sequence_dataset import SequenceDataset


class Uniclust30Dataset(SequenceDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
    ):
        super().__init__(root)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
