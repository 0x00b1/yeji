import pytest
import torch
from torch import Size
from yeji.features import TaitBryanAngle


class TestTaitBryanAngles:
    @pytest.fixture
    def tait_bryan_angles(self) -> TaitBryanAngle:
        return TaitBryanAngle(torch.tensor([[1, 2, 3]]))

    def test___new__(self):
        result = TaitBryanAngle(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, TaitBryanAngle)

        assert result.shape == Size([1, 3])

        with pytest.raises(ValueError):
            TaitBryanAngle(torch.tensor(5))

        with pytest.raises(ValueError):
            TaitBryanAngle(torch.tensor([1, 2]))

    def test___repr__(self, tait_bryan_angles: TaitBryanAngle):
        assert isinstance(tait_bryan_angles.__repr__(), str)

    def test__wrap(self):
        result = TaitBryanAngle._wrap(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, TaitBryanAngle)

        assert result.shape == Size([1, 3])

    def test_wrap_like(self, tait_bryan_angles: TaitBryanAngle):
        result = TaitBryanAngle.wrap_like(
            tait_bryan_angles,
            torch.tensor([[1, 2, 3]]),
        )

        assert isinstance(result, TaitBryanAngle)

        assert result.shape == Size([1, 3])
