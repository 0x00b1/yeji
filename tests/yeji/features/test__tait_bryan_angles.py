import pytest
import torch
from torch import Size
from yeji.features import TaitBryanAngles


class TestTaitBryanAngles:
    @pytest.fixture
    def tait_bryan_angles(self) -> TaitBryanAngles:
        return TaitBryanAngles(torch.tensor([[1, 2, 3]]))

    def test___new__(self):
        result = TaitBryanAngles(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, TaitBryanAngles)

        assert result.shape == Size([1, 3])

        with pytest.raises(ValueError):
            TaitBryanAngles(torch.tensor(5))

        with pytest.raises(ValueError):
            TaitBryanAngles(torch.tensor([1, 2]))

    def test___repr__(self, tait_bryan_angles: TaitBryanAngles):
        assert isinstance(tait_bryan_angles.__repr__(), str)

    def test__wrap(self):
        result = TaitBryanAngles._wrap(torch.tensor([[1, 2, 3]]))

        assert isinstance(result, TaitBryanAngles)

        assert result.shape == Size([1, 3])

    def test_wrap_like(self, tait_bryan_angles: TaitBryanAngles):
        result = TaitBryanAngles.wrap_like(
            tait_bryan_angles,
            torch.tensor([[1, 2, 3]]),
        )

        assert isinstance(result, TaitBryanAngles)

        assert result.shape == Size([1, 3])
