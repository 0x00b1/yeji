from torch import Tensor


def rotation_vector_magnitude(
    input: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    """
    Rotation vector magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, magnitudes are assumed to be in degrees. Default, `False`.

    Returns
    -------
    rotation_vector_magnitudes : Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError
