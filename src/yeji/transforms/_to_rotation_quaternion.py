from typing import Any, Dict

from torch import Tensor

import yeji.transforms.functional
from yeji.features import RotationQuaternions

from ._transform import Transform


class ToRotationQuaternion(Transform):
    _transformed_types = (Tensor,)

    def _transform(
        self,
        input: Tensor,
        parameters: Dict[str, Any],
    ) -> RotationQuaternions:
        return yeji.transforms.functional.to_rotation_quaternions(input)
