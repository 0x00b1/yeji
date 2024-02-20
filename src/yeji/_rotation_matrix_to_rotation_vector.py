from torch import Tensor


def rotation_matrix_to_rotation_vector(input: Tensor) -> Tensor:
    """
    Convert rotation matrices to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    rotation_vectors : Tensor, shape (..., 3)
        Rotation vectors.
    """
    raise NotImplementedError
