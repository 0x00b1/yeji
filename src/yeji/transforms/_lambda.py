from typing import Callable, Any, Type, Dict

from ._transform import Transform


class Lambda(Transform):
    _transformed_types = (object,)

    def __init__(self, fn: Callable, *types: Type):
        super().__init__()

        self._fn = fn

        self.types = types or self._transformed_types

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, self.types):
            return self._fn(inpt)

        return inpt

    def extra_repr(self) -> str:
        extras = []

        name = getattr(self._fn, "__name__", None)

        if name:
            extras.append(name)

        types_ = [type.__name__ for type in self.types]

        extras.append(f"types={types_}")

        return ", ".join(extras)
