from torch import Tensor


def tait_bryan_angle_to_rotation_vector(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Tait-Bryan angles to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Tait-Bryan angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    rotation_vectors : Tensor, shape (..., 3)
        Rotation vectors.
    """
    raise NotImplementedError
