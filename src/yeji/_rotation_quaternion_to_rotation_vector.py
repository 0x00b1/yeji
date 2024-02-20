from torch import Tensor


def rotation_quaternion_to_rotation_vector(input: Tensor) -> Tensor:
    """
    Convert rotation quaternions to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    Returns
    -------
    rotation_vectors : Tensor, shape (..., 3)
        Rotation vectors.
    """
    raise NotImplementedError
