from pathlib import Path
from typing import Any, Union

from torch.utils.data import Dataset


class Tox21Dataset(Dataset):
    def __init__(self, root: Union[str, Path]) -> None:
        super().__init__()

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
