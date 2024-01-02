import pytest
import torch
from torch import Size
from yeji.features import RotationQuaternion


class TestRotationQuaternion:
    @pytest.fixture
    def rotation_quaternion(self):
        return RotationQuaternion(torch.tensor([0, 1, 0, 0]))

    def test___new__(self):
        result = RotationQuaternion(torch.tensor([0, 1, 0, 0]))

        assert isinstance(result, RotationQuaternion)

        assert result.shape == Size([1, 4])

        with pytest.raises(ValueError):
            RotationQuaternion(torch.tensor([1, 2, 3]))

    def test___repr__(self, rotation_quaternion):
        assert isinstance(rotation_quaternion.__repr__(), str)

    def test__wrap(self):
        result = RotationQuaternion._wrap(torch.tensor([0, 1, 0, 0]))

        assert isinstance(result, RotationQuaternion)

        assert result.shape == Size([4])

    def test_wrap_like(self, rotation_quaternion):
        result = RotationQuaternion.wrap_like(
            rotation_quaternion,
            torch.tensor([1, 0, 0, 0]),
        )

        assert isinstance(result, RotationQuaternion)

        assert result.shape == Size([4])
