from pathlib import Path
from typing import Callable, Optional, Union

import yeji.io
from yeji.transforms import Transform

from ._lmdb_dataset import LMDBDataset


class ATOM3DDataset(LMDBDataset):
    def __init__(
        self,
        root: Union[str, Path],
        path: Union[str, Path],
        resource: str,
        name: str,
        *,
        checksum: Optional[str] = None,
        download: bool = False,
        transform_fn: Union[Callable, Transform, None] = None,
    ):
        r"""

        Parameters
        ----------
        root : str | Path
            Root directory of dataset.

        download: bool
            If `True`, downloads the dataset to the root directory. If dataset
            already exists, it is not redownloaded. Default, `False`.

        transform_fn : Callable | Transform | None
            Transforms the input.

        target_transform_fn : Callable | Transform | None
            Transforms the target.
        """
        self._root = root

        if isinstance(self._root, str):
            self._root = Path(self._root).resolve()

        self._transform_fn = transform_fn

        if download:
            yeji.io.download_and_extract_archive(
                resource,
                self._root / f"ATOM3D{name}",
                checksum=checksum,
            )

        super().__init__(
            self._root / f"ATOM3D{name}" / path,
            transform_fn=transform_fn,
        )
