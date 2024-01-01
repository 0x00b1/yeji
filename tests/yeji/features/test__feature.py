import copy

import pytest
import torch

from yeji.features._feature import _Feature
import yeji.transforms.functional


class Test_Feature:
    def test__to_tensor(self, feature: _Feature):
        result = _Feature._to_tensor([1, 2, 3])

        assert torch.is_tensor(result)

        assert not result.requires_grad

    def test_wrap_like(self):
        with pytest.raises(NotImplementedError):
            _Feature.wrap_like(None, None)

    def test___torch_function__(self, feature: _Feature):
        result = feature.__torch_function__(
            torch.add,
            (_Feature, torch.Tensor),
            (feature, torch.tensor([1, 2, 3])),
        )

        assert torch.is_tensor(result)

        assert not isinstance(result, _Feature)

    def test__f(self, feature: _Feature):
        assert feature._f == yeji.transforms.functional

    def test_device(self, feature: _Feature):
        assert feature.device == feature.device

    def test_ndim(self, feature: _Feature):
        assert feature.ndim == 1

    def test_dtype(self, feature: _Feature):
        assert feature.dtype == torch.int64

    def test_shape(self, feature: _Feature):
        assert feature.shape == (3,)

    def test___deepcopy__(self, feature: _Feature):
        with pytest.raises(NotImplementedError):
            copy.deepcopy(feature)
