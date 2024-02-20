from torch import Tensor


def invert_rotation_vector(
    input: Tensor,
    degrees: bool = False,
) -> Tensor:
    """
    Invert rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    inverted_rotation_vectors : Tensor, shape (..., 3)
        Inverted rotation vectors.
    """
    raise NotImplementedError
