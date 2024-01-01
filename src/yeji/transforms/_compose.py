from typing import Any, Callable, Sequence

from ._transform import Transform


class Compose(Transform):
    """
    Composes transforms.

    Args:
        transforms (sequence of ``Transform`` objects): sequence of transforms
        to compose.

    Example
    -------
    >>> from yeji.transforms import Compose, Identity
    >>>
    >>> Compose([
    >>>     Identity(),
    >>>     Identity(),
    >>>     Identity(),
    >>> ])

    Note
    ----
    Use ``torch.nn.Sequential`` to compile transforms.

        >>> from torch.nn import Sequential
        >>> import torch
        >>>
        >>> from yeji.transforms import Identity
        >>>
        >>> transforms = Sequential(
        >>>     Identity(),
        >>>     Identity(),
        >>> )
        >>>
        >>> compiled_transforms = torch.compile(transforms)

    Ensure only compilable transforms are used.
    """

    def __init__(self, transforms: Sequence[Callable]):
        """
        Parameters
        ----------
        transforms : Sequence[Callable]
            A sequence of callable objects representing the transforms to be
            composed.

        Raises
        ------
        TypeError
            If the ``transforms`` parameter is not of type ``Sequence``.

        ValueError
            If the ``transforms`` sequence is empty.

        Attributes
        ----------
        _transforms : Sequence[Callable]
            The sequence of callable objects representing the composed
            transforms.
        """
        super().__init__()

        if not isinstance(transforms, Sequence):
            raise TypeError

        if not transforms:
            raise ValueError

        self._transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        """
        Parameters
        ----------
        inputs : Sequence[Any]
            The input data to be transformed.

        Returns
        -------
        Any
            The transformed output data.

        """
        needs_unpacking = len(inputs) > 1

        ys = []

        for transform in self._transforms:
            ys = transform(*inputs)

            if needs_unpacking:
                inputs = ys
            else:
                inputs = (ys,)

        return ys

    def extra_repr(self) -> str:
        """
        Return a string representation of the `Compose` object.

        Returns
        -------
        str
            The string representation of the current `Compose` object,
            including all transforms.

        """
        string = []

        for transform in self._transforms:
            string.append(f"    {transform}")

        return "\n".join(string)
