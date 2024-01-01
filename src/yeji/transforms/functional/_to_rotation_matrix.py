from torch import Tensor

from yeji.features import RotationMatrix


def to_rotation_matrix(input: Tensor) -> RotationMatrix:
    return input
