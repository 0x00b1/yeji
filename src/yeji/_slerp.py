import torch
import torch.testing
from torch import Tensor


def slerp(
    input: Tensor,
    time: Tensor,
    rotation: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor:
    r"""
    Interpolate between two or more points on a sphere.

    Unlike linear interpolation, which can result in changes in speed when
    interpolating between orientations or positions on a sphere, spherical
    linear interpolation ensures that the interpolation occurs at a constant
    rate and follows the shortest path on the surface of the sphere.
    The process is useful for rotations and orientation interpolation in
    three-dimensional spaces, smoothly transitioning between different
    orientations.

    Mathematically, spherical linear interpolation interpolates between two
    points on a sphere using a parameter $t$, where $t = 0$ represents the
    start point and $t = n$ represents the end point. For two rotation
    quaternions $q_{1}$ and $q_{2}$ representing the start and end
    orientations:

    $$\text{slerp}(q_{1}, q_{2}; t) = q_{1}\frac{\sin((1 - t)\theta)}{\sin(\theta)} + q_{2}\frac{\sin(t\theta)}{\sin(\theta)}$$

    where $\theta$ is the angle between $q_{1}$ and $q_{2}$, and is computed
    using the dot product of $q_{1}$ and $q_{2}$. This formula ensures that the
    interpolation moves along the shortest path on the four-dimensional sphere
    of rotation quaternions, resulting in a smooth and constant-speed rotation.

    Parameters
    ----------
    input : Tensor, shape (..., N)
        Times.

    time : Tensor, shape (..., N)
        Times of the known rotations. At least 2 times must be specified.

    rotation : Tensor, shape (..., N, 4)
        Rotation quaternions. Rotation quaternions are normalized to unit norm.

    out : Tensor
        Output.
    """
    if time.shape[-1] != rotation.shape[-2]:
        raise ValueError("`times` and `rotations` must match in size.")

    interpolated_rotation_quaternions = torch.empty(
        [*input.shape, 4],
        dtype=input.dtype,
    )

    for index, t in enumerate(input):
        # FIND KEYFRAMES IN INTERVAL, ASSUME `times` IS SORTED:
        b = torch.min(torch.nonzero(torch.greater_equal(time, t)))

        if b > 0:
            a = b - 1
        else:
            a = b

        # IF `times` MATCHES A KEYFRAME:
        if time[b] == t or b == a:
            interpolated_rotation_quaternions[index] = rotation[b]

            continue

        time_0, time_1 = time[a], time[b]

        relative_time = torch.divide(
            torch.subtract(
                t,
                time_0,
            ),
            torch.subtract(
                time_1,
                time_0,
            ),
        )

        quaternion_0, quaternion_1 = rotation[a], rotation[b]

        cos_theta = torch.dot(quaternion_0, quaternion_1)

        # NOTE: IF `cos_theta` IS NEGATIVE, `rotations` HAS OPPOSITE HANDEDNESS
        # SO `slerp` WON’T TAKE THE SHORTER PATH. INVERT `quaternion_1`.
        #       @0X00B1, FEBRUARY, 26, 2024
        if cos_theta < 0.0:
            quaternion_1 = torch.negative(quaternion_1)

            cos_theta = torch.negative(cos_theta)

        # USE LINEAR INTERPOLATION IF QUATERNIONS ARE CLOSE:
        if cos_theta > 0.9995:
            interpolated_rotation_quaternion = torch.add(
                torch.multiply(
                    torch.subtract(
                        torch.tensor(1.0),
                        relative_time,
                    ),
                    quaternion_0,
                ),
                torch.multiply(
                    relative_time,
                    quaternion_1,
                ),
            )
        else:
            interpolated_rotation_quaternion = torch.add(
                torch.multiply(
                    quaternion_0,
                    torch.divide(
                        torch.sin(
                            torch.multiply(
                                torch.subtract(
                                    torch.tensor(1.0),
                                    relative_time,
                                ),
                                torch.atan2(
                                    torch.sqrt(
                                        torch.subtract(
                                            torch.tensor(1.0),
                                            torch.square(cos_theta),
                                        )
                                    ),
                                    cos_theta,
                                ),
                            ),
                        ),
                        torch.sqrt(
                            torch.subtract(
                                torch.tensor(1.0),
                                torch.square(cos_theta),
                            )
                        ),
                    ),
                ),
                torch.multiply(
                    quaternion_1,
                    torch.divide(
                        torch.sin(
                            torch.multiply(
                                relative_time,
                                torch.atan2(
                                    torch.sqrt(
                                        torch.subtract(
                                            torch.tensor(1.0),
                                            torch.square(cos_theta),
                                        )
                                    ),
                                    cos_theta,
                                ),
                            ),
                        ),
                        torch.sqrt(
                            torch.subtract(
                                torch.tensor(1.0),
                                torch.square(cos_theta),
                            ),
                        ),
                    ),
                ),
            )

        interpolated_rotation_quaternions[index] = torch.divide(
            interpolated_rotation_quaternion,
            torch.norm(interpolated_rotation_quaternion),
        )

    return interpolated_rotation_quaternions
