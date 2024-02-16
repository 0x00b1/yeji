from torch import Tensor

from yeji.features import RotationQuaternion


def to_rotation_quaternion(input: Tensor) -> RotationQuaternion:
    return input
