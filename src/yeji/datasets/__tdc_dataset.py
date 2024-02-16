from pathlib import Path

import pandas
import pooch
from torch.utils.data import Dataset


class _TDCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        *,
        identifier: int,
        suffix: str,
        checksum: str,
        x_columns: list[str],
        y_columns: list[str],
    ):
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        if download:
            pooch.retrieve(
                f"https://dataverse.harvard.edu/api/access/datafile/{identifier}",
                fname=f"{self.__class__.__name__}.{suffix}",
                known_hash=checksum,
                path=root / self.__class__.__name__,
            )

        path = (
            root
            / self.__class__.__name__
            / f"{self.__class__.__name__}.{suffix}"
        )

        match path.suffix:
            case ".csv":
                self.data = pandas.read_csv(path)
            case ".pkl":
                self.data = pandas.read_pickle(path)
            case ".tsv":
                self.data = pandas.read_csv(path, sep="\t")
            case _:
                raise ValueError

        self.xs = self.data[x_columns].apply(tuple, axis=1)
        self.ys = self.data[y_columns].apply(tuple, axis=1)

    def __getitem__(self, index: int):
        x = self.xs[index]

        if len(x) == 1:
            x = x[0]

        y = self.ys[index]

        if len(y) == 1:
            y = y[0]

        return x, y

    def __len__(self) -> int:
        return len(self.data)
