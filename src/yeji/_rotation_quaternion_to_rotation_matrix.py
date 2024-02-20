from torch import Tensor


def rotation_quaternion_to_rotation_matrix(input: Tensor) -> Tensor:
    """
    Convert rotation quaternions to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    Returns
    -------
    rotation_matrices : Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    raise NotImplementedError
