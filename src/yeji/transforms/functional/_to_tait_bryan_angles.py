from torch import Tensor

from yeji.features import TaitBryanAngles


def to_tait_bryan_angles(input: Tensor) -> TaitBryanAngles:
    return input
