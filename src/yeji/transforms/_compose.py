from typing import Any, Callable, Sequence

from ._transform import Transform


class Compose(Transform):
    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__()

        if not isinstance(transforms, Sequence):
            raise TypeError(
                "Argument transforms should be a sequence of callables"
            )
        elif not transforms:
            raise ValueError("Pass at least one transform")

        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        for transform in self.transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        format_string = []

        for t in self.transforms:
            format_string.append(f"    {t}")

        return "\n".join(format_string)
