from typing import Any, Dict

from ._transform import Transform


class Identity(Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt
