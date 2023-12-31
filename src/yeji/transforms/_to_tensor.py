from typing import Any, Dict

from torch import Tensor

from ..features._feature import Feature
from ._transform import Transform


class ToTensor(Transform):
    _transformed_types = (Feature,)

    def _transform(
        self,
        input: Feature,
        parameters: Dict[str, Any],
    ) -> Tensor:
        return input.as_subclass(Tensor)
