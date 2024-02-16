from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor

from ._feature import Feature


class RotationVector(Feature):
    degrees: bool

    @classmethod
    def _wrap(cls, tensor: Tensor, *, degrees: bool = False) -> RotationVector:
        if tensor.ndim == 0:
            raise ValueError

        if tensor.shape[-1] != 3:
            raise ValueError

        if tensor.ndim == 1:
            tensor = torch.unsqueeze(tensor, 0)

        rotation_vector = tensor.as_subclass(cls)

        rotation_vector.degrees = degrees

        return rotation_vector

    def __new__(
        cls,
        data: Any,
        *,
        degrees: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> RotationVector:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        return cls._wrap(tensor, degrees=degrees)

    @classmethod
    def wrap_like(
        cls,
        other: RotationVector,
        tensor: Tensor,
    ) -> RotationVector:
        return cls._wrap(tensor)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(degrees=self.degrees)


_RotationVectorType = Union[Tensor, RotationVector]
_RotationVectorTypeJIT = Tensor

_TensorRotationVectorType = Union[Tensor, RotationVector]
_TensorRotationVectorTypeJIT = Tensor
