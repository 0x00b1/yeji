from typing import Any, Dict

from torch import Tensor

import yeji.transforms.functional
from yeji.features import EulerAngles

from ._transform import Transform


class ToEulerAngles(Transform):
    _transformed_types = (Tensor,)

    def _transform(
        self,
        input: Tensor,
        parameters: Dict[str, Any],
    ) -> EulerAngles:
        return yeji.transforms.functional.to_euler_angles(input)
