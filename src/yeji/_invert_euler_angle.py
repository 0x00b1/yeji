from torch import Tensor


def invert_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool | None = False,
) -> Tensor:
    """
    Invert Euler angles.

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
    inverted_euler_angles : Tensor, shape (..., 3)
        Inverted Euler angles.
    """
    raise NotImplementedError
