from torch import Tensor


def rotation_matrix_magnitude(input: Tensor) -> Tensor:
    """
    Rotation matrix magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    rotation_matrix_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError
