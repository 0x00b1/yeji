from typing import Any, Dict

from torch import Tensor

from ._transform import Transform
from ..features._feature import _Feature


class ToTensor(Transform):
    _transformed_types = (_Feature,)

    def _transform(
        self,
        input: _Feature,
        parameters: Dict[str, Any],
    ) -> Tensor:
        return input.as_subclass(Tensor)
