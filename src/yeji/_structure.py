# @title Operators

import torch
from torch import Generator, Tensor


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


def apply_rotation_matrix_to_vector(
    input: Tensor,
    rotation: Tensor,
    inverse: bool | None = False,
) -> Tensor:
    """
    Apply rotation matrices to a set of vectors.

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
        of rotation matrices and number of vectors must follow standard
        broadcasting rules: either one of them equals unity or they both equal
        each other.

    rotation : Tensor, shape (..., 3, 3)
        Rotation matrices.

    inverse : bool, optional
        If `True` the inverse of the rotation matrices are applied to the input
        vectors. Default, `False`.

    Returns
    -------
    rotated_vectors : Tensor, shape (..., 3)
        Rotated vectors.
    """
    raise NotImplementedError


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


def apply_rotation_vector_to_vector(
    input: Tensor,
    rotation: Tensor,
    inverse: bool | None = False,
    degrees: bool | None = False,
) -> Tensor:
    """
    Apply rotation vectors to a set of vectors.

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
        of rotation vectors and number of vectors must follow standard
        broadcasting rules: either one of them equals unity or they both equal
        each other.

    rotation : Tensor, shape (..., 4)
        Rotation vectors.

    inverse : bool, optional
        If `True` the inverse of the rotation vectors are applied to the input
        vectors. Default, `False`.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    rotated_vectors : Tensor, shape (..., 3)
        Rotated vectors.
    """
    raise NotImplementedError


def apply_tait_bryan_angle_to_vector(
    input: Tensor,
    rotation: Tensor,
    axes: str,
    degrees: bool = False,
    inverse: bool | None = False,
) -> Tensor:
    """
    Apply Tait-Bryan angles to a set of vectors.

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


def euler_angle_identity(
    size: int,
    axes: str,
    degrees: bool | None = False,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    """
    Identity Euler angles.

    Parameters
    ----------
    size : int
        Output size.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    Returns
    -------
    identity_euler_angles : Tensor, shape (size, 3)
        Identity Euler angles.
    """
    raise NotImplementedError


def euler_angle_magnitude(
    input: Tensor,
    axes: str,
    degrees: bool | None = False,
) -> Tensor:
    """
    Euler angle magnitudes.

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
    euler_angle_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError


def euler_angle_to_rotation_matrix(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Euler angles to rotation matrices.

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
    rotation_matrices : Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    raise NotImplementedError


def euler_angle_to_rotation_quaternion(
    input: Tensor,
    axes: str,
    degrees: bool = False,
    canonical: bool | None = False,
) -> Tensor:
    """
    Convert Euler angles to rotation quaternions.

    Parameters
    ----------
    input : Tensor
        Euler angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    rotation_quaternions : Tensor, shape (..., 4)
        Rotation quaternions.
    """
    raise NotImplementedError


def euler_angle_to_rotation_vector(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Euler angles to rotation vectors.

    Parameters
    ----------
    input : Tensor
        Euler angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees and returned
        rotation vector magnitudes are in degrees. Default, `False`.

    Returns
    -------
    rotation_vectors : Tensor, shape (..., 3)
        Rotation vectors.
    """
    raise NotImplementedError


def euler_angle_to_tait_bryan_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Euler angles to Tait-Bryan angles.

    Parameters
    ----------
    input : Tensor
        Euler angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees and returned
        Tait-Bryan angles are in degrees. Default, `False`.

    Returns
    -------
    tait_bryan_angles : Tensor, shape (..., 3)
        Tait-Bryan angles.
    """
    raise NotImplementedError


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


def invert_rotation_matrix(input: Tensor) -> Tensor:
    """
    Invert rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    inverted_rotation_matrices : Tensor, shape (..., 3, 3)
        Inverted rotation matrices.
    """
    raise NotImplementedError


def invert_rotation_quaternion(
    input: Tensor,
    canonical: bool = False,
) -> Tensor:
    """
    Invert rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    inverted_rotation_quaternions : Tensor, shape (..., 4)
        Inverted rotation quaternions.
    """
    raise NotImplementedError


def invert_rotation_vector(
    input: Tensor,
    degrees: bool = False,
) -> Tensor:
    """
    Invert rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    inverted_rotation_vectors : Tensor, shape (..., 3)
        Inverted rotation vectors.
    """
    raise NotImplementedError


def invert_tait_bryan_angles(
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


def random_euler_angle(
    size: int,
    axes: str,
    degrees: bool | None = False,
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate random Euler angles.

    Parameters
    ----------
    size : int
        Output size.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    generator : torch.Generator, optional
        Psuedo-random number generator. Default, `None`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    pin_memory : bool, optional
        If `True`, returned tensor is allocated in pinned memory. Default,
        `False`.

    Returns
    -------
    random_euler_angles : Tensor, shape (..., 3)
        Random Euler angles.

        The returned Euler angles are in the range:

            *   First angle: :math:`(-180, 180]` degrees (inclusive)
            *   Second angle:
                    *   :math:`[-90, 90]` degrees if all axes are different
                        (e.g., :math:`xyz`)
                    *   :math:`[0, 180]` degrees if first and third axes are
                        the same (e.g., :math:`zxz`)
            *   Third angle: :math:`[-180, 180]` degrees (inclusive)
    """
    raise NotImplementedError


def random_rotation_matrix(
    size: int,
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate random rotation matrices.

    Parameters
    ----------
    size : int
        Output size.

    generator : torch.Generator, optional
        Psuedo-random number generator. Default, `None`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    pin_memory : bool, optional
        If `True`, returned tensor is allocated in pinned memory. Default,
        `False`.

    Returns
    -------
    random_rotation_matrices : Tensor, shape (..., 3, 3)
        Random rotation matrices.
    """
    raise NotImplementedError


def random_rotation_quaternion(
    size: int,
    canonical: bool = False,
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate random rotation quaternions.

    Parameters
    ----------
    size : int
        Output size.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    generator : torch.Generator, optional
        Psuedo-random number generator. Default, `None`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    pin_memory : bool, optional
        If `True`, returned tensor is allocated in pinned memory. Default,
        `False`.

    Returns
    -------
    random_rotation_quaternions : Tensor, shape (..., 4)
        Random rotation quaternions.
    """
    raise NotImplementedError


def random_rotation_vector(
    size: int,
    degrees: bool = False,
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate random rotation vectors.

    Parameters
    ----------
    size : int
        Output size.

    degrees

    generator : torch.Generator, optional
        Psuedo-random number generator. Default, `None`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    pin_memory : bool, optional
        If `True`, returned tensor is allocated in pinned memory. Default,
        `False`.

    Returns
    -------
    random_rotation_vectors : Tensor, shape (..., 3)
        Random rotation vectors.
    """
    raise NotImplementedError


def random_tait_bryan_angle(
    size: int,
    axes: str,
    degrees: bool | None = False,
    *,
    generator: Generator | None = None,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
    pin_memory: bool | None = False,
) -> Tensor:
    """
    Generate random Tait-Bryan angles.

    Parameters
    ----------
    size : int
        Output size.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    generator : torch.Generator, optional
        Psuedo-random number generator. Default, `None`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    pin_memory : bool, optional
        If `True`, returned tensor is allocated in pinned memory. Default,
        `False`.

    Returns
    -------
    random_tait_bryan_angles : Tensor, shape (..., 3)
        Random Tait-Bryan angles.
    """
    raise NotImplementedError


def rotation_matrix_identity(
    size: int,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    """
    Identity rotation matrices.

    Parameters
    ----------
    size : int
        Output size.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    Returns
    -------
    identity_rotation_matrices : Tensor, shape (size, 3, 3)
        Identity rotation matrices.
    """
    raise NotImplementedError


def rotation_matrix_magnitude(input: Tensor) -> Tensor:
    """
    Rotation matrix magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    rotation_matrix_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError


def rotation_matrix_to_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert rotation matrices to Euler angles.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

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


def rotation_matrix_to_rotation_quaternion(
    input: Tensor,
    canonical: bool | None = False,
) -> Tensor:
    """
    Convert rotation matrices to rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    rotation_quaternions : Tensor, shape (..., 4)
        Rotation quaternions.
    """
    raise NotImplementedError


def rotation_matrix_to_rotation_vector(input: Tensor) -> Tensor:
    """
    Convert rotation matrices to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    rotation_vectors : Tensor, shape (..., 3)
        Rotation vectors.
    """
    raise NotImplementedError


def rotation_matrix_to_tait_bryan_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert rotation matrices to Tait-Bryan angles.

    Parameters
    ----------
    input : Tensor, shape (..., 3, 3)
        Rotation matrices.

    Returns
    -------
    tait_bryan_angles : Tensor, shape (..., 3)
        Tait-Bryan angles.
    """
    raise NotImplementedError


def rotation_quaternion_identity(
    size: int,
    canonical: bool | None = False,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    """
    Identity rotation quaternions.

    Parameters
    ----------
    size : int
        Output size.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    Returns
    -------
    identity_rotation_quaternions : Tensor, shape (size, 4)
        Identity rotation quaternions.
    """
    raise NotImplementedError


def rotation_quaternion_magnitude(input: Tensor) -> Tensor:
    """
    Rotation quaternion magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions.

    Returns
    -------
    rotation_quaternion_magnitudes: Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError


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


def rotation_quaternion_to_rotation_matrix(input: Tensor) -> Tensor:
    """
    Convert rotation quaternions to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    Returns
    -------
    rotation_matrices : Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    raise NotImplementedError


def rotation_quaternion_to_rotation_vector(input: Tensor) -> Tensor:
    """
    Convert rotation quaternions to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    Returns
    -------
    rotation_vectors : Tensor, shape (..., 3)
        Rotation vectors.
    """
    raise NotImplementedError


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


def rotation_vector_identity(
    size: int,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    """
    Identity rotation vectors.

    Parameters
    ----------
    size : int
        Output size.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    Returns
    -------
    identity_rotation_vectors : Tensor, shape (size, 3)
        Identity rotation vectors.
    """
    raise NotImplementedError


def rotation_vector_magnitude(
    input: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    """
    Rotation vector magnitudes.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, magnitudes are assumed to be in degrees. Default, `False`.

    Returns
    -------
    rotation_vector_magnitudes : Tensor, shape (...)
        Angles in radians. Magnitudes will be in the range :math:`[0, \pi]`.
    """
    raise NotImplementedError


def rotation_vector_to_euler_angle(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert rotation vectors to Euler angles.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

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


def rotation_vector_to_rotation_matrix(
    input: Tensor,
    degrees: bool | None = False,
) -> Tensor:
    """
    Convert rotation vectors to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    Returns
    -------
    rotation_matrices : Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    raise NotImplementedError


def rotation_vector_to_rotation_quaternion(
    input: Tensor,
    degrees: bool | None = False,
    canonical: bool | None = False,
) -> Tensor:
    """
    Convert rotation vectors to rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Rotation vectors.

    degrees : bool, optional
        If `True`, rotation vector magnitudes are assumed to be in degrees.
        Default, `False`.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    rotation_quaternions : Tensor, shape (..., 4)
        Rotation quaternions.
    """
    raise NotImplementedError


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


def tait_bryan_angles_identity(
    size: int,
    axes: str,
    degrees: bool | None = False,
    *,
    out: Tensor | None = None,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool | None = False,
) -> Tensor:
    """
    Identity Tait-Bryan angles.

    Parameters
    ----------
    size : int
        Output size.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    out : Tensor, optional
        Output tensor. Default, `None`.

    dtype : torch.dtype, optional
        Type of the returned tensor. Default, global default.

    layout : torch.layout, optional
        Layout of the returned tensor. Default, `torch.strided`.

    device : torch.device, optional
        Device of the returned tensor. Default, current device for the default
        tensor type.

    requires_grad : bool, optional
        Whether autograd records operations on the returned tensor. Default,
        `False`.

    Returns
    -------
    identity_tait_bryan_angles : Tensor, shape (size, 3)
        Identity Tait-Bryan angles.
    """
    raise NotImplementedError


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


def tait_bryan_angle_to_rotation_matrix(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Tait-Bryan angles to rotation matrices.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Tait-Bryan angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    rotation_matrices : Tensor, shape (..., 3, 3)
        Rotation matrices.
    """
    raise NotImplementedError


def tait_bryan_angle_to_rotation_quaternion(
    input: Tensor,
    axes: str,
    degrees: bool = False,
    canonical: bool | None = False,
) -> Tensor:
    """
    Convert Tait-Bryan angles to rotation quaternions.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Tait-Bryan angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    canonical : bool, optional
        Whether to map the redundant double cover of rotation space to a unique
        canonical single cover. If `True`, then the rotation quaternion is
        chosen from :math:`{q, -q}` such that the :math:`w` term is positive.
        If the :math:`w` term is :math:`0`, then the rotation quaternion is
        chosen such that the first non-zero term of the :math:`x`, :math:`y`,
        and :math:`z` terms is positive.

    Returns
    -------
    rotation_quaternions : Tensor, shape (..., 4)
        Rotation quaternions.
    """
    raise NotImplementedError


def tait_bryan_angle_to_rotation_vector(
    input: Tensor,
    axes: str,
    degrees: bool = False,
) -> Tensor:
    """
    Convert Tait-Bryan angles to rotation vectors.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        Tait-Bryan angles.

    axes : str
        Axes. 1-3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic
        rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations. Extrinsic and
        intrinsic rotations cannot be mixed.

    degrees : bool, optional
        If `True`, Euler angles are assumed to be in degrees. Default, `False`.

    Returns
    -------
    rotation_vectors : Tensor, shape (..., 3)
        Rotation vectors.
    """
    raise NotImplementedError
