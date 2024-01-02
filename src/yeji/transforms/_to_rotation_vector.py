from typing import Dict, Any

from torch import Tensor

import yeji.transforms.functional
from yeji.features import RotationVector
from ._transform import Transform


class ToRotationVector(Transform):
    _transformed_types = (Tensor,)

    def _transform(
        self,
        input: Tensor,
        parameters: Dict[str, Any],
    ) -> RotationVector:
        return yeji.transforms.functional.to_rotation_vector(input)
