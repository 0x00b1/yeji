from torch import Tensor

from yeji.features import EulerAngles


def to_euler_angles(
    input: Tensor,
    axes: str = "xyz",
    degrees: bool = False,
) -> EulerAngles:
    return input
