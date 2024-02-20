from torch import Tensor


def rotation_quaternion_magnitude(input: Tensor) -> Tensor:
    """
    Rotation quaternion magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions.

    Returns
    -------
    rotation_quaternion_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError
