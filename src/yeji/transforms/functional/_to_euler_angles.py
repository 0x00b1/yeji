from torch import Tensor

from yeji.features import EulerAngles


def to_euler_angles(input: Tensor) -> EulerAngles:
    return input
