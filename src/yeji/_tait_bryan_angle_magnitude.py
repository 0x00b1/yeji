from torch import Tensor


def tait_bryan_angle_magnitude(
    input: Tensor,
    axes: str,
    degrees: bool | None = False,
) -> Tensor:
    """
    Magnitude of Tait-Bryan angles.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Tait-Bryan angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Tait-Bryan angles are assumed to be in degrees. Default,
        `False`.

    Returns
    -------
    tait_bryan_angle_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError
