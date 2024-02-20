from torch import Tensor


def rotation_quaternion_to_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert rotation quaternions to Euler angles.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    Returns
    -------
    euler_angles : Tensor, shape (..., 3)
        Euler angles. The returned Euler angles are in the range:

            * First angle: :math:`(-180, 180]` degrees (inclusive)
            * Second angle:
                * :math:`[-90, 90]` degrees if all axes are different
                  (e.g., :math:`xyz`)
                * :math:`[0, 180]` degrees if first and third axes are the same
                  (e.g., :math:`zxz`)
            * Third angle: :math:`[-180, 180]` degrees (inclusive)
    """
    raise NotImplementedError
