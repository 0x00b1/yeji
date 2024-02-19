from torch import Tensor

import yeji.nn.functional
from yeji.features import (
    EulerAngle,
    RotationMatrix,
    RotationQuaternion,
    RotationVector,
    TaitBryanAngle,
)


def to_euler_angle(
    input: Tensor,
    axes: str = "xyz",
    degrees: bool = False,
) -> EulerAngle:
    match input:
        case RotationMatrix():
            return EulerAngle(
                yeji.nn.functional.rotation_matrix_to_euler_angle(
                    input,
                    axes=axes,
                    degrees=degrees,
                ),
                axes=axes,
                degrees=degrees,
            )
        case RotationQuaternion():
            return EulerAngle(
                yeji.nn.functional.rotation_quaternion_to_euler_angle(
                    input,
                    axes=axes,
                    degrees=degrees,
                ),
                axes=axes,
                degrees=degrees,
            )
        case RotationVector():
            return EulerAngle(
                yeji.nn.functional.rotation_vector_to_euler_angle(
                    input,
                    axes=axes,
                    degrees=degrees,
                ),
                axes=axes,
                degrees=degrees,
            )
        case TaitBryanAngle():
            return EulerAngle(
                yeji.nn.functional.tait_bryan_angle_to_euler_angle(
                    input,
                    axes=axes,
                    degrees=degrees,
                ),
                axes=axes,
                degrees=degrees,
            )
        case _:
            raise ValueError
