from torch import Tensor


def euler_angle_to_rotation_matrix(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Euler angles to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Euler angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    rotation_matrices : Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    raise NotImplementedError
