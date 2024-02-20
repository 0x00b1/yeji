from torch import Tensor


def euler_angle_to_tait_bryan_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Euler angles to Tait-Bryan angles.

    Parameters
    ----------
    input : Tensor
        Euler angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees and returned
        Tait-Bryan angles are in degrees. Default, `False`.

    Returns
    -------
    tait_bryan_angles : Tensor, shape (..., 3)
        Tait-Bryan angles.
    """
    raise NotImplementedError
