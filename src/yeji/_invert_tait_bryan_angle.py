from torch import Tensor


def invert_tait_bryan_angle(
    input: Tensor,
    axes: str,
    degrees: bool | None = False,
) -> Tensor:
    """
    Invert Tait-Bryan angles.

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
    inverted_tait_bryan_angles : Tensor, shape (..., 3)
        Inverted Tait-Bryan angles.
    """
    raise NotImplementedError
