from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor

from ._feature import Feature


class EulerAngles(Feature):
    axes: str
    degrees: bool

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        axes: str = "xyz",
        degrees: bool = False,
    ) -> EulerAngles:
        if tensor.ndim == 0:
            raise ValueError

        if tensor.shape[-1] != 3:
            raise ValueError

        if tensor.ndim == 1:
            tensor = torch.unsqueeze(tensor, 0)

        euler_angles = tensor.as_subclass(cls)

        euler_angles.axes = axes

        euler_angles.degrees = degrees

        return euler_angles

    def __new__(
        cls,
        data: Any,
        *,
        axes: str = "xyz",
        degrees: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> EulerAngles:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        return cls._wrap(tensor, axes=axes, degrees=degrees)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(axes=self.axes, degrees=self.degrees)


_EulerAnglesType = Union[Tensor, EulerAngles]
_EulerAnglesTypeJIT = Tensor

_TensorEulerAnglesType = Union[Tensor, EulerAngles]
_TensorEulerAnglesTypeJIT = Tensor
