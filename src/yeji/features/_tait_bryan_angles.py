from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor

from yeji.features._feature import Feature


class TaitBryanAngles(Feature):
    @classmethod
    def _wrap(cls, tensor: Tensor) -> TaitBryanAngles:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> TaitBryanAngles:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if tensor.ndim == 0:
            raise ValueError

        if tensor.shape[-1] != 3:
            raise ValueError

        if tensor.ndim == 1:
            tensor = torch.unsqueeze(tensor, 0)

        return cls._wrap(tensor)

    @classmethod
    def wrap_like(
        cls,
        other: TaitBryanAngles,
        tensor: Tensor,
    ) -> TaitBryanAngles:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


_TaitBryanAnglesType = Union[Tensor, TaitBryanAngles]
_TaitBryanAnglesTypeJIT = Tensor

_TensorTaitBryanAnglesType = Union[Tensor, TaitBryanAngles]
_TensorTaitBryanAnglesTypeJIT = Tensor
