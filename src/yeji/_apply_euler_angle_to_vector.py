from torch import Tensor


def apply_euler_angle_to_vector(
    input: Tensor,
    rotation: Tensor,
    axes: str,
    degrees: bool = False,
    inverse: bool | None = False,
) -> Tensor:
    """
    Apply Euler angles to a set of vectors.

    If the original frame rotates to the final frame by this rotation, then its
    application to a vector can be seen in two ways:

        1.  As a projection of vector components expressed in the final frame
            to the original frame.
        2.  As the physical rotation of a vector being glued to the original
            frame as it rotates. In this case the vector components are
            expressed in the original frame before and after the rotation.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Each vector represents a vector in three-dimensional space. The number
        of Euler angles and number of vectors must follow standard broadcasting
        rules: either one of them equals unity or they both equal each other.

    rotation : Tensor, shape (..., 3)
        Euler angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    inverse : bool, optional
        If `True` the inverse of the Euler angles are applied to the input
        vectors. Default, `False`.

    Returns
    -------
    rotated_vectors : Tensor, shape (..., 3)
        Rotated vectors.
    """
    raise NotImplementedError
