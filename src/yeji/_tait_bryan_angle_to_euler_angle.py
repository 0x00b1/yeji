from torch import Tensor


def tait_bryan_angle_to_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Tait-Bryan angles to Euler angles.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Tait-Bryan angles.

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
