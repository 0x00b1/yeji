from typing import Any, Dict

from torch import Tensor

import yeji.transforms.functional
from yeji.features import TaitBryanAngle

from ._transform import Transform


class ToTaitBryanAngle(Transform):
    _transformed_types = (Tensor,)

    def _transform(
        self,
        input: Tensor,
        parameters: Dict[str, Any],
    ) -> TaitBryanAngle:
        return yeji.transforms.functional.to_tait_bryan_angles(input)
