import hypothesis
import hypothesis.strategies
import numpy
import torch
from scipy.spatial.transform import Rotation
from yeji.nn.functional import rotation_quaternions_to_rotation_matrix


@hypothesis.given(
    hypothesis.strategies.lists(
        hypothesis.strategies.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1,
            max_value=1,
        ),
        min_size=4,
        max_size=4,
    )
    .filter(lambda x: numpy.linalg.norm(x) > 0.1)
    .map(numpy.array)
)
def test_rotation_quaternions_to_rotation_matrix(quaternion):
    # Convert the quaternion to a scipy rotation object and then to a matrix
    scipy_rot = Rotation.from_quat(quaternion)
    scipy_matrix = scipy_rot.as_matrix()

    # Convert the quaternion to a PyTorch tensor and use our function
    torch_quaternion = torch.tensor(quaternion, dtype=torch.float32)
    torch_matrix = rotation_quaternions_to_rotation_matrix(
        torch_quaternion.unsqueeze(0),
    )[0].numpy()

    # Check if the matrices are approximately equal
    numpy.testing.assert_allclose(torch_matrix, scipy_matrix, rtol=1e-5)
