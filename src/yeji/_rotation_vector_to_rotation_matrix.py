from torch import Tensor


def rotation_vector_to_rotation_matrix(
    input: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    """
    Convert rotation vectors to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    rotation_matrices : Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    raise NotImplementedError
