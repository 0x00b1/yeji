import torch
from torch import Generator, Tensor


def random_rotation_quaternion(
    size: int,
    canonical: bool = False,
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
    Generate random rotation quaternions.

    Parameters
    ----------
    size : int
        Output size.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

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
    random_rotation_quaternions : Tensor, shape (..., 4)
        Random rotation quaternions.
    """
    raise NotImplementedError
