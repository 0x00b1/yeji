from torch import Tensor


def apply_tait_bryan_angle_to_vector(
    input: Tensor,
    rotation: Tensor,
    axes: str,
    degrees: bool = False,
    inverse: bool | None = False,
) -> Tensor:
    r"""
    Rotates vectors in three-dimensional space using Tait-Bryan angles.

    Note
    ----
    This function interprets the rotation of the original frame to the final
    frame as either a projection, where it maps the components of vectors from
    the final frame to the original frame, or as a physical rotation,
    integrating the vectors into the original frame during the rotation
    process. Consequently, the vector components are maintained in the original
    frame’s perspective both before and after the rotation.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Each vector represents a vector in three-dimensional space. The number
        of Tait-Bryan angles and number of vectors must follow standard
        broadcasting rules: either one of them equals unity or they both equal
        each other.

    rotation : Tensor, shape (..., 3)
        Tait-Bryan angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Tait-Bryan angles are assumed to be in degrees. Default,
        `False`.

    inverse : bool, optional
        If `True` the inverse of the Tait-Bryan angles are applied to the input
        vectors. Default, `False`.

    Returns
    -------
    rotated_vectors : Tensor, shape (..., 3)
        Rotated vectors.
    """
    raise NotImplementedError
