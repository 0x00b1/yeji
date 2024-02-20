from torch import Tensor


def rotation_vector_to_rotation_quaternion(
    input: Tensor,
    degrees: bool | None = False,
    canonical: bool | None = False,
) -> Tensor:
    """
    Convert rotation vectors to rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    rotation_quaternions : Tensor, shape (..., 4)
        Rotation quaternions.
    """
    raise NotImplementedError
