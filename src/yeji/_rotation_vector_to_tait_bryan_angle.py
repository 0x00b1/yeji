from torch import Tensor


def rotation_vector_to_tait_bryan_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert rotation vectors to Tait-Bryan angles.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    tait_bryan_angles : Tensor, shape (..., 3)
        Tait-Bryan angles.
    """
    raise NotImplementedError
