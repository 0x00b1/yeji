from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Sequence, TypeVar

from lightning import LightningDataModule
from torch.utils.data import Sampler

from yeji.transforms import Transform

T = TypeVar("T")


class AqSolDBSolubilityLightningDataModule(LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        *,
        download: bool = False,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
        lengths: Sequence[float] | None = None,
        generator: Generator | None = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Iterable | Sampler | None = None,
        batch_sampler: Iterable[Sequence] | Sampler[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[Sequence[T]], Any] | None = None,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        r"""

        Parameters
        ----------
        root : str | Path
            Root directory where the dataset subdirectory exists or, if
            `download` is `True`, the directory where the dataset subdirectory
            will be created and the dataset downloaded.

        download : bool
            If `True`, download the dataset and to the :attr:`root` directory
            (default: `False`). If the dataset is already downloaded, it is not
            redownloaded.

        transform_fn : Callable | Transform | None
            A `Callable` that maps a sequence to a transformed sequence
            (default: `None`).

        target_transform_fn : Callable | Transform | None
            `Callable` that maps a target to a transformed target
            (default: `None`).

        lengths : Sequence[float] | None
            Fractions of splits to generate.

        generator : Generator | None
            Generator used for the random permutation (default: `None`).

        seed : int
            Desired seed. Value must be within the inclusive range
            `[-0x8000000000000000, 0xFFFFFFFFFFFFFFFF]`
            (default: `0xDEADBEEF`). Otherwise, a `RuntimeError` is raised.
            Negative inputs are remapped to positive values with the formula
            `0xFFFFFFFFFFFFFFFF + seed`.

        batch_size : int
            Samples per batch (default: `1`).

        shuffle : bool
            If `True`, reshuffle datasets at every epoch (default: `True`).

        sampler : Iterable | Sampler | None
            Strategy to draw samples from the dataset (default: `None`). Can be
            any `Iterable` with `__len__` implemented. If specified, `shuffle`
            must be `False`.

        batch_sampler : Iterable[Sequence] | Sampler[Sequence] | None
            `sampler`, but returns a batch of indices (default: `None`).
            Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and
            `drop_last`.

        num_workers : int
            Subprocesses to use (default: `0`). `0` means that the datasets
            will be loaded in the main process.

        collate_fn : Callable[[Sequence[T]], Any] | None
            Merges samples to form a mini-batch of Tensor(s) (default: `None`).

        pin_memory : bool
            If `True`, Tensors are copied to the device’s (e.g., CUDA) pinned
            memory before returning them (default: `True`).

        drop_last : bool
            If `True`, drop the last incomplete batch, if the dataset size is
            not divisible by the batch size (default: `False`). If `False` and
            the size of dataset is not divisible by the batch size, then the
            last batch will be smaller.
        """
        super().__init__()
