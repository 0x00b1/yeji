import pytest
import torch
from torch import Size

from yeji.features import EulerAngles


class TestEulerAngles:
    @pytest.fixture
    def euler_angles(self) -> EulerAngles:
        return EulerAngles(torch.tensor([[1, 2, 3]]))

    def test___new__(self):
        result = EulerAngles(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, EulerAngles)

        assert result.shape == Size([1, 3])

        with pytest.raises(ValueError):
            EulerAngles(torch.tensor(5))

        with pytest.raises(ValueError):
            EulerAngles(torch.tensor([1, 2]))

    def test___repr__(self, euler_angles: EulerAngles):
        assert isinstance(euler_angles.__repr__(), str)

    def test__wrap(self):
        result = EulerAngles._wrap(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, EulerAngles)

        assert result.shape == Size([1, 3])

    def test_wrap_like(self, euler_angles: EulerAngles):
        result = EulerAngles.wrap_like(
            euler_angles,
            torch.tensor([[1, 2, 3]]),
        )

        assert isinstance(result, EulerAngles)

        assert result.shape == Size([1, 3])
