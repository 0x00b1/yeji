from torch import Tensor


def rotation_matrix_to_tait_bryan_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert rotation matrices to Tait-Bryan angles.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    tait_bryan_angles : Tensor, shape (..., 3)
        Tait-Bryan angles.
    """
    raise NotImplementedError
