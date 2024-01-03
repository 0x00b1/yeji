import torch
from torch import Tensor


def rotation_quaternions_to_rotation_matrix(input: Tensor) -> Tensor:
    n = input.shape[0]

    w, x, y, z = input[:, 0], input[:, 1], input[:, 2], input[:, 3]

    matrices = torch.zeros((n, 3, 3))

    matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)

    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    matrices[:, 1, 2] = 2 * (y * z - x * w)

    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return matrices
