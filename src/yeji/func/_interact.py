# @title `prescient.func.interact`

import dataclasses
import functools
from enum import Enum
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import optree
import torch
from optree import PyTree
from torch import Tensor

from ._space import (
    _canonicalize_distance_fn,
    _dataclass,
    _map_product,
    _safe_mask,
    _safe_sum,
    _zero_diagonal_mask,
)


@_dataclass
class ParameterTree:
    tree: PyTree
    kind: "ParameterTreeKind" = dataclasses.field(metadata={"static": True})


class ParameterTreeKind(Enum):
    BOND = 0
    KINDS = 1
    PARTICLE = 2
    SPACE = 3


def _bonded_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    static_bonds: Optional[Tensor] = None,
    static_kinds: Optional[Tensor] = None,
    ignore_unused_parameters: bool = False,
    **static_kwargs,
) -> Callable[..., Tensor]:
    merge_dictionaries_fn = functools.partial(
        _merge_dictionaries,
        ignore_unused_parameters=ignore_unused_parameters,
    )

    def mapped_fn(
        positions: Tensor,
        bonds: Optional[Tensor] = None,
        kinds: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        accumulator = torch.tensor(
            0.0,
            device=positions.device,
            dtype=positions.dtype,
        )

        distance_fn = functools.partial(displacement_fn, **kwargs)

        distance_fn = torch.func.vmap(distance_fn, 0, 0)

        if bonds is not None:
            parameters = merge_dictionaries_fn(static_kwargs, kwargs)

            for name, parameter in parameters.items():
                if kinds is not None:
                    parameters[name] = _to_bond_kind_parameters(
                        parameter,
                        kinds,
                    )

            interactions = distance_fn(
                positions[bonds[:, 0]],
                positions[bonds[:, 1]],
            )

            interactions = _safe_sum(fn(interactions, **parameters))

            accumulator = accumulator + interactions

        if static_bonds is not None:
            parameters = merge_dictionaries_fn(static_kwargs, kwargs)

            for name, parameter in parameters.items():
                if static_kinds is not None:
                    parameters[name] = _to_bond_kind_parameters(
                        parameter,
                        static_kinds,
                    )

            interactions = distance_fn(
                positions[static_bonds[:, 0]],
                positions[static_bonds[:, 1]],
            )

            interactions = _safe_sum(fn(interactions, **parameters))

            accumulator = accumulator + interactions

        return accumulator

    return mapped_fn


def _kwargs_to_pairwise_interaction_parameters(
    kwargs: Dict[str, Union["ParameterTree", Tensor, float, PyTree]],
    combinators: Dict[str, Callable],
    kinds: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    parameters = {}

    for name, parameter in kwargs.items():
        if kinds is None:

            def _combinator_fn(x: Tensor, y: Tensor) -> Tensor:
                return (x + y) * 0.5

            combinator = combinators.get(name, _combinator_fn)

            match parameter:
                case ParameterTree():
                    match parameter.kind:
                        case ParameterTreeKind.BOND | ParameterTreeKind.SPACE:
                            parameters[name] = parameter.tree
                        case ParameterTreeKind.PARTICLE:

                            def _particle_fn(_parameter: Tensor) -> Tensor:
                                return combinator(
                                    _parameter[:, None, ...],
                                    _parameter[None, :, ...],
                                )

                            parameters[name] = optree.tree_map(
                                _particle_fn,
                                parameter.tree,
                            )
                        case _:
                            message = f"""
parameter `kind` is `{parameter.kind}`. If `kinds` is `None` and a parameter is
an instance of `ParameterTree`, `kind` must be `ParameterTreeKind.BOND`,
`ParameterTreeKind.PARTICLE`, or `ParameterTreeKind.SPACE`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case Tensor():
                    match parameter.ndim:
                        case 0 | 2:
                            parameters[name] = parameter
                        case 1:
                            parameters[name] = combinator(
                                parameter[:, None],
                                parameter[None, :],
                            )
                        case _:
                            message = f"""
parameter `ndim` is `{parameter.ndim}`. If `kinds` is `None` and a parameter is
an instance of `Tensor`, `ndim` must be in `0`, `1`, or `2`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case float() | int():
                    parameters[name] = parameter
                case _:
                    message = f"""
parameter `type` is {type(parameter)}. If `kinds` is `None`, a parameter must
be an instance of `ParameterTree`, `Tensor`, `float`, or `int`.
                    """.replace("\n", " ")

                    raise ValueError(message)
        else:
            if name in combinators:
                raise ValueError

            match parameter:
                case ParameterTree():
                    match parameter.kind:
                        case ParameterTreeKind.SPACE:
                            parameters[name] = parameter.tree
                        case ParameterTreeKind.KINDS:

                            def _kinds_fn(_parameter: Tensor) -> Tensor:
                                return _parameter[kinds]

                            parameters[name] = optree.tree_map(
                                _kinds_fn,
                                parameter.tree,
                            )
                        case _:
                            message = f"""
parameter `kind` is {parameter.kind}. If `kinds` is `None` and a parameter is
an instance of `ParameterTree`, `kind` must be `ParameterTreeKind.SPACE` or
`ParameterTreeKind.KINDS`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case Tensor():
                    match parameter.ndim:
                        case 0:
                            parameters[name] = parameter
                        case 2:
                            parameters[name] = parameter[kinds]
                        case _:
                            message = f"""
parameter `ndim` is `{parameter.ndim}`. If `kinds` is not `None` and a
parameter is an instance of `Tensor`, `ndim` must be in `0`, `1`, or `2`.
                            """.replace("\n", " ")

                            raise ValueError(message)
                case _:
                    parameters[name] = parameter

    return parameters


def _merge_dictionaries(
    this: Dict,
    that: Dict,
    ignore_unused_parameters: bool = False,
):
    if not ignore_unused_parameters:
        return {**this, **that}

    merged_dictionaries = dict(this)

    for this_key in merged_dictionaries.keys():
        that_value = that.get(this_key)

        if that_value is not None:
            merged_dictionaries[this_key] = that_value

    return merged_dictionaries


def _pairwise_interaction(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    kinds: Optional[Union[int, Tensor]] = None,
    dim: Optional[Tuple[int, ...]] = None,
    keepdim: bool = False,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Tensor]:
    parameters, combinators = {}, {}

    for name, parameter in kwargs.items():
        if isinstance(parameter, Callable):
            combinators[name] = parameter
        elif isinstance(parameter, tuple) and isinstance(
            parameter[0], Callable
        ):
            assert len(parameter) == 2

            combinators[name], parameters[name] = parameter[0], parameter[1]
        else:
            parameters[name] = parameter

    merge_dicts = functools.partial(
        _merge_dictionaries,
        ignore_unused_parameters=ignore_unused_parameters,
    )

    if kinds is None:

        def mapped_fn(_position: Tensor, **_dynamic_kwargs) -> Tensor:
            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distances = _map_product(distance_fn)(_position, _position)

            dictionaries = merge_dicts(parameters, _dynamic_kwargs)

            to_parameters = _kwargs_to_pairwise_interaction_parameters(
                dictionaries,
                combinators,
            )

            u = fn(distances, **to_parameters)

            u = _zero_diagonal_mask(u)

            u = _safe_sum(u, dim=dim, keepdim=keepdim)

            return u * 0.5

        return mapped_fn

    if isinstance(kinds, Tensor):
        if not isinstance(kinds, Tensor) or kinds.is_floating_point():
            raise ValueError

        kinds_count = int(torch.max(kinds))

        if dim is not None or keepdim:
            raise ValueError

        def mapped_fn(_position: Tensor, **_dynamic_kwargs):
            u = torch.tensor(0.0, dtype=torch.float32)

            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distance_fn = _map_product(distance_fn)

            for m in range(kinds_count + 1):
                for n in range(m, kinds_count + 1):
                    distance = distance_fn(
                        _position[kinds == m],
                        _position[kinds == n],
                    )

                    _kwargs = merge_dicts(parameters, _dynamic_kwargs)

                    s_kwargs = _kwargs_to_pairwise_interaction_parameters(
                        _kwargs, combinators, (m, n)
                    )

                    u = fn(distance, **s_kwargs)

                    if m == n:
                        u = _zero_diagonal_mask(u)

                        u = _safe_sum(u)

                        u = u + u * 0.5
                    else:
                        y = _safe_sum(u)

                        u = u + y

            return u

        return mapped_fn

    if isinstance(kinds, int):
        kinds_count = kinds

        def mapped_fn(_position: Tensor, _kinds: Tensor, **_dynamic_kwargs):
            if not isinstance(_kinds, Tensor) or _kinds.is_floating_point():
                raise ValueError

            u = torch.tensor(0.0, dtype=torch.float32)

            n = _position.shape[0]

            distance_fn = functools.partial(displacement_fn, **_dynamic_kwargs)

            distance_fn = _map_product(distance_fn)

            _kwargs = merge_dicts(parameters, _dynamic_kwargs)

            distance = distance_fn(_position, _position)

            for m in range(kinds_count):
                for n in range(kinds_count):
                    a = torch.reshape(
                        _kinds == m,
                        [
                            n,
                        ],
                    )
                    b = torch.reshape(
                        _kinds == n,
                        [
                            n,
                        ],
                    )

                    a = a.to(dtype=_position.dtype)[:, None]
                    b = b.to(dtype=_position.dtype)[None, :]

                    mask = a * b

                    if m == n:
                        mask = _zero_diagonal_mask(mask) * mask

                    to_parameters = _kwargs_to_pairwise_interaction_parameters(
                        _kwargs, combinators, (m, n)
                    )

                    y = fn(distance, **to_parameters) * mask

                    y = _safe_sum(y, dim=dim, keepdim=keepdim)

                    u = u + y

            return u / 2.0

        return mapped_fn

    raise ValueError


def _soft_sphere(
    distances: Tensor,
    diameter: float | int | Tensor = 1.0,
    scale: float | int | Tensor = 1.0,
    stiffness: float | int | Tensor = 2.0,
    **_,
) -> Tensor:
    distances = distances / diameter

    def _fn(dr: Tensor) -> Tensor:
        return scale / stiffness * (1.0 - dr) ** stiffness

    if isinstance(stiffness, int) or (
        isinstance(distances, Tensor)
        and not (distances.is_complex() or distances.is_floating_point())
    ):
        return torch.where(torch.less(distances, 1.0), _fn(distances), 0.0)

    return _safe_mask(distances < 1.0, _fn, distances, 0.0)


def _spring(
    distances: Tensor,
    elastic_response: Tensor = 2,
    equilibrium_distance: Tensor = 1,
    stiffness: Tensor = 1,
    **_,
) -> Tensor:
    return (
        stiffness
        / elastic_response
        * torch.abs(distances - equilibrium_distance) ** elastic_response
    )


def _to_bond_kind_parameters(
    parameter: Tensor | ParameterTree,
    kinds: Tensor,
) -> Tensor | ParameterTree:
    assert isinstance(kinds, Tensor)

    assert len(kinds.shape) == 1

    match parameter:
        case Tensor():
            match parameter.shape:
                case 0:
                    return parameter
                case 1:
                    return parameter[kinds]
                case _:
                    raise ValueError
        case ParameterTree():
            if parameter.kind is ParameterTreeKind.BOND:

                def _fn(_parameter: Dict) -> Tensor:
                    return _parameter[kinds]

                return optree.tree_map(_fn, parameter.tree)

            if parameter.kind is ParameterTreeKind.SPACE:
                return parameter.tree

            raise ValueError
        case float() | int():
            return parameter
        case _:
            raise NotImplementedError


def interact(
    fn: Callable[..., Tensor],
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    interaction: Literal[
        "angle",
        "bond",
        "dihedral",
        "long-range",
        "neighbor",
        "pair",
        "triplet",
    ],
    *,
    bonds: Optional[Tensor] = None,
    kinds: Optional[Tensor] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    ignore_unused_parameters: bool = False,
    **kwargs,
) -> Callable[..., Tensor]:
    r"""
    Define interactions between elements of a system.

    For a collection of $N$ elements, $\vec{r}_i \in \mathbb{R}^{D}$, where $1 \leq i \leq N$, energy is the function $U : \mathbb{R}^{N \times D} \rightarrow \mathbb{R}$. Energy is used by a simulation by applying Newton's laws: $m \vec{\ddot{r}}_{i} = - \nabla_{\vec{r}_{i}} U$ where $m$ is mass. Rather than defining an energy as an interaction between all the elements in the simulation space simultaneously, it's preferable to use a pairwise energy function based on the displacement between a pair of elements, $u(\vec{r}_{i} - \vec{r}_{j})$. Total energy is defined by the sum over pairwise interactions:

    $$U = \frac{1}{2} \sum_{i \neq j} u(\vec{r}_{i} - \vec{r}_{j}).$$

    To facilitate the construction of functions from interactions, `interact` returns a function to map bonds, neighbors, pairs, or triplets interactions and transforms them to operate on an entire simulation.

    Parameters
    ----------
    fn : Callable[..., Array]
        Function that takes distances or displacements of shape `(n, m)` or `(n, m, spatial_dimension)` and `kwargs` and returns values of shape `(n, m, spatial_dimension)`. The function must be a differentiable function as the force is computed using automatic differentiation (see `prescient.func.force`).

    displacement_fn : Callable[[Tensor, Tensor], Tensor]
        Displacement function that takes positions of shape `(spatial_dimension)` and `(spatial_dimension)` and returns distances or displacements of shape `()` or `(spatial_dimension)`.

    interaction : Literal["bond", "neighbor", "pair", "triplet"]
        One of the following types of interactions:

        -   `"angle"`,

        -   `"bond"`, transforms a function that acts on a single pair of elements to a function that acts on a set of bonds.

        -  `"dihedral"`,

        -   `"long-range"`,

        -   `"neighbor"`, transforms a function that acts on pairs of elements to a function that acts on neighbor lists.

        -   `"pair"`, transforms a function that acts on a pair of elements to a function that acts on a system of interacting elements.

        -   `"triplet"`, transforms a function that acts on triplets of elements to a function that acts on a system of interacting elements. Many common empirical potentials include three-body terms, this type of pairwise interaction simplifies the loss computation by transforming a loss function that acts on two pairwise displacements or distances to a loss function that acts on a system of interacting elements.

    bonds : Optional[Tensor], default=None

    kinds : Optional[Tensor], default=None
        Kinds for the different elements. Should either be `None` (in which case it is assumed that all the elements have the same kind) or labels of shape `(n)`. If `intraction` is `"pair"` or `"triplet"`, kinds can be dynamically specified by passing the `kinds` keyword argument to the mapped function.

    dim : Optional[Union[int, Tuple[int, ...]]], default=None
        Dimension or dimensions to reduce. If `None`, all dimensions are reduced.

    keepdim : bool, default=False
        Whether the output has `dim` retained or not.

    ignore_unused_parameters : bool, default=True

    kwargs :
        `kwargs` passed to the function. Depends on the `interaction` type:

        *   If `interaction` is `"bond"` and `kinds` is `None`, must be a scalar or a tensor of shape `(n)`. If `interaction` is `"bond"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"neighbor"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)`, tensor of shape `(n, n)`, a PyTree of parameters and corresponding mapping, or a binary function that determines how per-element parameters are combined. If `kinds` is `None`, `kinds` is defined as the average of the two per-element parameters. If `interaction` is `"neighbor"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds, kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"pair"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)`, tensor of shape `(n, n)`, a PyTree of parameters and corresponding mapping, or a binary function that determines how per-element parameters are combined. If `kinds` is `None`, `kinds` is defined as the average of the two per-element parameters. If `interaction` is `"pairwise"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds, kinds)`, or a PyTree of parameters and corresponding mapping.

        *   If `interaction` is `"triplet"` and `kinds` is `None`, must be a scalar, tensor of shape `(n)` based on the central element, or a tensor of shape `(n, n, n)` defining triplet interactions. If `interaction` is `"triplet"` and `kinds` is not `None`, must be a scalar, a tensor of shape `(kinds)`, or a tensor of shape `(kinds, kinds, kinds)` defining triplet interactions.

    Returns
    -------
    : Callable[..., Tensor]
        Signature of the return function depends on `interaction`:

        *   `"bond"`:
                `(positions, bonds, kinds) -> Tensor`

            The return function can optionally take the keyword arguments `bonds` and `kinds` to dynamically allocate bonds.

        *   `"neighbor"`:
                `(positions, neighbors) -> Tensor`

            The return function takes positions of shape `(n, spatial_dimension)` and neighbor labels of shape `(n, neighbors)`.

        *   `"pair"`:
                `(positions, kinds, maximum_kind, **kwargs) -> Tensor`

            If `kinds` is `None` or static, the return function takes positions of shape `(n, spatial_dimension)`. If `kinds` is dynamic, the return function takes positions of shape `(n, spatial_dimension)`, integer labels of shape (n), and an integer specifying the maximum kind. The return function can optionally take keyword arguments to pass to the displacement function.

        *   `"triplet"`:
                `(positions, kinds, maximum_kind, **kwargs) -> Tensor`

            If `kinds` is `None` or static, the return function takes positions of shape `(n, spatial_dimension)`. If `kinds` is dynamic, the return function takes positions of shape `(n, spatial_dimension)`, integer labels of shape (n), and an integer specifying the maximum kind. The return function can optionally take keyword arguments to pass to the displacement function.

    Examples
    --------
    Create a pairwise interaction from a potential function:

        def fn(x: Tensor, a: float, e: float, s: float, **_) -> Tensor:
            return e / a * (1.0 - x / s) ** a

        displacement_fn, _ = prescient.func.space([10.0], parallelpiped=False)

        fn = prescient.func.interact(
            fn,
            displacement_fn,
            interaction="pair",
            a=2.0,
            e=1.0,
            s=1.0,
        )
    """
    match interaction:
        case "bonded":
            return _bonded_interaction(
                fn,
                displacement_fn,
                static_bonds=bonds,
                static_kinds=kinds,
                ignore_unused_parameters=ignore_unused_parameters,
                **kwargs,
            )
        case "neighbor":
            raise NotImplementedError
        case "pairwise":
            return _pairwise_interaction(
                fn,
                displacement_fn,
                kinds=kinds,
                dim=dim,
                keepdim=keepdim,
                ignore_unused_parameters=ignore_unused_parameters,
                **kwargs,
            )
        case "triplet":
            raise NotImplementedError


def soft_sphere(
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    kinds: Optional[Tensor] = None,
    diameter: float | int | Tensor = 1.0,
    scale: float | int | Tensor = 1.0,
    stiffness: float | int | Tensor = 2.0,
    per_particle: bool = False,
) -> Callable[[Tensor], Tensor]:
    if per_particle:
        dim = [
            1,
        ]
    else:
        dim = None

    return interact(
        _soft_sphere,
        _canonicalize_distance_fn(displacement_fn),
        interaction="pairwise",
        kinds=kinds,
        dim=dim,
        ignore_unused_parameters=True,
        diameter=diameter,
        scale=scale,
        stiffness=stiffness,
    )


def spring(
    displacement_fn: Callable[[Tensor, Tensor], Tensor],
    bonds: Tensor,
    kinds: Optional[Tensor] = None,
    elastic_response: Tensor = 2,
    equilibrium_distance: Tensor = 1,
    stiffness: Tensor = 1,
) -> Callable[[Tensor], Tensor]:
    return interact(
        _spring,
        _canonicalize_distance_fn(displacement_fn),
        interaction="bonded",
        bonds=bonds,
        kinds=kinds,
        ignore_unused_parameters=True,
        elastic_response=elastic_response,
        equilibrium_distance=equilibrium_distance,
        stiffness=stiffness,
    )
