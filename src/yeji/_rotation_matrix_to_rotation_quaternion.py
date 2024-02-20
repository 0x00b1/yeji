from torch import Tensor


def rotation_matrix_to_rotation_quaternion(
    input: Tensor,
    canonical: bool | None = False,
) -> Tensor:
    """
    Convert rotation matrices to rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

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
