from pathlib import Path
from typing import Callable, Optional, Union

from ._uniref_dataset import UniRefDataset


class UniRef50Dataset(UniRefDataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        cache_sequence_indicies: bool = True,
        download: bool = False,
        transform_fn: Optional[Callable] = None,
        target_transform_fn: Optional[Callable] = None,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param cache_sequence_indicies: If ``True``, caches the sequence
            indicies to disk for faster re-initialization (default: ``True``).

        :param download: If ``True``, download the dataset and to the
            :attr:`root` directory (default: ``False``). If the dataset is
            already downloaded, it is not redownloaded.

        :param transform_fn: A ``Callable`` that maps a sequence to a
            transformed sequence (default: ``None``).

        :param target_transform_fn: ``Callable`` that maps a target (a cluster
            identifier) to a transformed target (default: ``None``).
        """
        super().__init__(
            root,
            "uniref50",
            (
                "2b05bd43c14ce0bc0591a017efa648e6",  # uniref50.fasta
                "cb28f2fc41694ccc009cbbab3e08db98",  # uniref50.fasta.gz
            ),
            cache_sequence_indicies=cache_sequence_indicies,
            download=download,
            transform_fn=transform_fn,
            target_transform_fn=target_transform_fn,
        )
