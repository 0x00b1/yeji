from torch import Tensor


def rotation_quaternion_to_tait_bryan_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert rotation quaternions to Tait-Bryan angles.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    Returns
    -------
    tait_bryan_angles : Tensor, shape (..., 3)
        Tait-Bryan angles.
    """
    raise NotImplementedError
