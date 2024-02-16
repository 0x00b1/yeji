from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor

from ._feature import Feature


class RotationMatrix(Feature):
    @classmethod
    def _wrap(cls, tensor: Tensor) -> RotationMatrix:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> RotationMatrix:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if tensor.ndim <= 1:
            raise ValueError

        if tensor.shape[-2:] == [3, 3]:
            raise ValueError

        if tensor.ndim == 2:
            tensor = torch.unsqueeze(tensor, 0)

        return cls._wrap(tensor)

    @classmethod
    def wrap_like(
        cls,
        other: RotationMatrix,
        tensor: Tensor,
    ) -> RotationMatrix:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


_RotationMatrixType = Union[Tensor, RotationMatrix]
_RotationMatrixTypeJIT = Tensor

_TensorRotationMatrixType = Union[Tensor, RotationMatrix]
_TensorRotationMatrixTypeJIT = Tensor
