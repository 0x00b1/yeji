import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation as R
from yeji.nn.functional import rotation_quaternions_to_rotation_matrix


@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=4,
    ).map(np.array)
)
def test_rotation_quaternions_to_rotation_matrix(quaternion):
    # Convert the quaternion to a scipy rotation object and then to a matrix
    scipy_rot = R.from_quat(quaternion)
    scipy_matrix = scipy_rot.as_matrix()

    # Convert the quaternion to a PyTorch tensor and use our function
    torch_quaternion = torch.tensor(quaternion, dtype=torch.float32)
    torch_matrix = rotation_quaternions_to_rotation_matrix(
        torch_quaternion.unsqueeze(0),
    )[0].numpy()

    # Check if the matrices are approximately equal
    np.testing.assert_allclose(torch_matrix, scipy_matrix, rtol=1e-5)
