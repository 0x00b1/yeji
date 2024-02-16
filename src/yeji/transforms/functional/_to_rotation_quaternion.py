from torch import Tensor

from yeji.features import RotationQuaternions


def to_rotation_quaternion(input: Tensor) -> RotationQuaternions:
    return input
