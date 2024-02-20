import torch
from torch import Generator, Tensor


def random_rotation_vector(
    size: int,
    degrees: bool = False,
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate random rotation vectors.

    Parameters
    ----------
    size : int
        Output size.

    degrees

    generator : torch.Generator, optional
        Psuedo-random number generator. Default, `None`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    pin_memory : bool, optional
        If `True`, returned tensor is allocated in pinned memory. Default,
        `False`.

    Returns
    -------
    random_rotation_vectors : Tensor, shape (..., 3)
        Random rotation vectors.
    """
    raise NotImplementedError
