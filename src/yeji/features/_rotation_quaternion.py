from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor

from ._feature import Feature


class RotationQuaternion(Feature):
    """
    Rotation quaternion.
    """

    @classmethod
    def _wrap(cls, tensor: Tensor) -> RotationQuaternion:
        return tensor.as_subclass(cls)

    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> RotationQuaternion:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        if tensor.shape[-1] != 4:
            raise ValueError

        if tensor.ndim == 1:
            tensor = torch.unsqueeze(tensor, 0)

        return cls._wrap(tensor)

    @classmethod
    def wrap_like(
        cls,
        other: RotationQuaternion,
        tensor: Tensor,
    ) -> RotationQuaternion:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr()


_RotationQuaternionType = Union[Tensor, RotationQuaternion]
_RotationQuaternionTypeJIT = Tensor

_TensorRotationQuaternionType = Union[Tensor, RotationQuaternion]
_TensorRotationQuaternionTypeJIT = Tensor
