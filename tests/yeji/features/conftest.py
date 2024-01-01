import pytest
import torch

from yeji.features._feature import _Feature


@pytest.fixture
def feature() -> _Feature:
    return _Feature(torch.tensor([1, 2, 3]))
