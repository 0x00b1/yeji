import pytest
import torch
from torch import Size
from yeji.features import RotationQuaternions


class TestRotationQuaternion:
    @pytest.fixture
    def rotation_quaternion(self):
        return RotationQuaternions(torch.tensor([0, 1, 0, 0]))

    def test___new__(self):
        result = RotationQuaternions(torch.tensor([0, 1, 0, 0]))

        assert isinstance(result, RotationQuaternions)

        assert result.shape == Size([1, 4])

        with pytest.raises(ValueError):
            RotationQuaternions(torch.tensor([1, 2, 3]))

    def test___repr__(self, rotation_quaternion):
        assert isinstance(rotation_quaternion.__repr__(), str)

    def test__wrap(self):
        result = RotationQuaternions._wrap(torch.tensor([0, 1, 0, 0]))

        assert isinstance(result, RotationQuaternions)

        assert result.shape == Size([4])

    def test_wrap_like(self, rotation_quaternion):
        result = RotationQuaternions.wrap_like(
            rotation_quaternion,
            torch.tensor([1, 0, 0, 0]),
        )

        assert isinstance(result, RotationQuaternions)

        assert result.shape == Size([4])
