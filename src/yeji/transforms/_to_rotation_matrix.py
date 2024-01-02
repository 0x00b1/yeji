from typing import Any, Dict

from torch import Tensor

import yeji.transforms.functional
from yeji.features import RotationMatrix

from ._transform import Transform


class ToRotationMatrix(Transform):
    _transformed_types = (Tensor,)

    def _transform(
        self,
        input: Tensor,
        parameters: Dict[str, Any],
    ) -> RotationMatrix:
        return yeji.transforms.functional.to_rotation_matrix(input)
