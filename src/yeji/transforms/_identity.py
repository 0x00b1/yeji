from typing import Any, Dict

from ._transform import Transform


class Identity(Transform):
    def _transform(self, input: Any, parameters: Dict[str, Any]) -> Any:
        return input
