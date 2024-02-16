from typing import Any, Dict

from torch import Tensor

import yeji.transforms.functional
from yeji.features import TaitBryanAngles

from ._transform import Transform


class ToTaitBryanAngle(Transform):
    _transformed_types = (Tensor,)

    def _transform(
        self,
        input: Tensor,
        parameters: Dict[str, Any],
    ) -> TaitBryanAngles:
        return yeji.transforms.functional.to_tait_bryan_angles(input)
