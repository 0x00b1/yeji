from torch import Tensor


def invert_rotation_matrix(input: Tensor) -> Tensor:
    """
    Invert rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    inverted_rotation_matrices : Tensor, shape (..., 3, 3)
        Inverted rotation matrices.
    """
    raise NotImplementedError
