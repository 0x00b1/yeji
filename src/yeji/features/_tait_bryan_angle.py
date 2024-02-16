from __future__ import annotations

from typing import Any, Optional, Union

import torch
from torch import Tensor

from ._feature import Feature


class TaitBryanAngle(Feature):
    axes: str
    degrees: bool

    @classmethod
    def _wrap(
        cls,
        tensor: Tensor,
        *,
        axes: str = "xyz",
        degrees: bool = False,
    ) -> TaitBryanAngle:
        if tensor.ndim == 0:
            raise ValueError

        if tensor.shape[-1] != 3:
            raise ValueError

        if tensor.ndim == 1:
            tensor = torch.unsqueeze(tensor, 0)

        tait_bryan_angle = tensor.as_subclass(cls)

        tait_bryan_angle.axes = axes

        tait_bryan_angle.degrees = degrees

        return tait_bryan_angle

    def __new__(
        cls,
        data: Any,
        *,
        axes: str = "xyz",
        degrees: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> TaitBryanAngle:
        tensor = cls._to_tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        return cls._wrap(tensor, axes=axes, degrees=degrees)

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return self._make_repr(axes=self.axes, degrees=self.degrees)


_TaitBryanAngleType = Union[Tensor, TaitBryanAngle]
_TaitBryanAngleTypeJIT = Tensor

_TensorTaitBryanAngleType = Union[Tensor, TaitBryanAngle]
_TensorTaitBryanAngleTypeJIT = Tensor
