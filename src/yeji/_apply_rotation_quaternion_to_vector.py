from torch import Tensor


def apply_rotation_quaternion_to_vector(
    input: Tensor,
    rotation: Tensor,
    inverse: bool | None = False,
) -> Tensor:
    """
    Apply rotation quaternions to a set of vectors.

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
        of rotation quaternions and number of vectors must follow standard
        broadcasting rules: either one of them equals unity or they both equal
        each other.

    rotation : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    inverse : bool, optional
        If `True` the inverse of the rotation quaternions are applied to the
        input vectors. Default, `False`.

    Returns
    -------
    rotated_vectors : Tensor, shape (..., 3)
        Rotated vectors.
    """
    raise NotImplementedError
