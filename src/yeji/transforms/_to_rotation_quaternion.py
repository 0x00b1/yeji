from typing import Dict, Any

from torch import Tensor

import yeji.transforms.functional
from yeji.features import EulerAngles
from ._transform import Transform


class ToRotationQuaternion(Transform):
    _transformed_types = (Tensor,)

    def _transform(
        self,
        input: Tensor,
        parameters: Dict[str, Any],
    ) -> EulerAngles:
        return yeji.transforms.functional.to_rotation_quaternion(input)
