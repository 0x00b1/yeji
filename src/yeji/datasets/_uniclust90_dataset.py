from pathlib import Path
from typing import Any, Union

from ._sequence_dataset import SequenceDataset


class Uniclust90Dataset(SequenceDataset):
    def __init__(self, root: Union[str, Path]) -> None:
        super().__init__(root)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
