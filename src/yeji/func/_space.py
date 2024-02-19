# @title Implementation

import dataclasses
import functools
import math
import operator
from enum import Enum
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import optree
import torch
from optree import PyTree
from torch import Tensor
from torch.autograd import Function

from ._force import force

YOSHIDA_SUZUKI = {
    1: [+1.0000000000000000],
    3: [+0.8289815435887510, -0.6579630871775020, +0.8289815435887510],
    5: [
        +0.2967324292201065,
        +0.2967324292201065,
        -0.1869297168804260,
        +0.2967324292201065,
        +0.2967324292201065,
    ],
    7: [
        +0.7845136104775600,
        +0.2355732133593570,
        -1.1776799841788700,
        +1.3151863206839100,
        -1.1776799841788700,
        +0.2355732133593570,
        +0.7845136104775600,
    ],
}

T = TypeVar("T")


def _dataclass(cls: Type[T]):
    def _set(self: dataclasses.dataclass, **kwargs):
        return dataclasses.replace(self, **kwargs)

    cls.set = _set

    dataclass_cls = dataclasses.dataclass(frozen=True)(cls)

    data_fields, metadata_fields = [], []

    for name, kind in dataclass_cls.__dataclass_fields__.items():
        if not kind.metadata.get("static", False):
            data_fields = [*data_fields, name]
        else:
            metadata_fields = [*metadata_fields, name]

    def _iterate_cls(_x) -> List[Tuple]:
        data_iterable = []

        for k in data_fields:
            data_iterable.append(getattr(_x, k))

        metadata_iterable = []

        for k in metadata_fields:
            metadata_iterable.append(getattr(_x, k))

        return [data_iterable, metadata_iterable]

    def _iterable_to_cls(meta, data):
        meta_args = tuple(zip(metadata_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)

        return dataclass_cls(**kwargs)

    optree.register_pytree_node(
        dataclass_cls,
        _iterate_cls,
        _iterable_to_cls,
        "prescient.func",
    )

    return dataclass_cls


def static_field():
    return dataclasses.field(metadata={"static": True})


class _ParameterTreeKind(Enum):
    BOND = 0
    PARTICLE = 1
    KINDS = 2
    SPACE = 3


@_dataclass
class _ParameterTree:
    tree: PyTree
    kind: _ParameterTreeKind = dataclasses.field(metadata={"static": True})


class _DispatchByState:
    def __init__(self, fn):
        self._fn = fn

        self._registry = {}

    def __call__(self, state, *args, **kwargs):
        if type(state.positions) in self._registry:
            return self._registry[type(state.positions)](
                state, *args, **kwargs
            )

        return self._fn(state, *args, **kwargs)

    def register(self, oftype):
        def register_fn(fn):
            self._registry[oftype] = fn

        return register_fn


@_dataclass
class _BrownianDynamicsState:
    positions: Tensor
    masses: Tensor


@_dataclass
class _FIREDescentOptimizerState:
    current_momentum: float
    forces: Tensor
    masses: Tensor
    momentums: Tensor
    positions: Tensor
    step_size: float
    steps: int


@_dataclass
class _NoseHooverChainState:
    degrees_of_freedom: int = dataclasses.field(metadata={"static": True})
    kinetic_energies: Tensor
    masses: Tensor
    momentums: Tensor
    oscillations: Tensor
    positions: Tensor


@_dataclass
class _NVEState:
    positions: Tensor
    momentums: Tensor
    forces: Tensor
    masses: Tensor

    @property
    def velocities(self) -> Tensor:
        return self.momentums / self.masses


@_dataclass
class _NVTLangevinThermostatState:
    forces: Tensor
    masses: Tensor
    momentums: Tensor
    positions: Tensor

    @property
    def velocities(self) -> Tensor:
        return self.momentums / self.masses


@_dataclass
class Normal:
    mean: Tensor
    var: Tensor

    def sample(self):
        mu, sigma = self.mean, torch.sqrt(self.var)

        return mu + sigma * torch.normal(0.0, 1.0, mu.shape, dtype=mu.dtype)

    def log_prob(self, x):
        return (
            -0.5 * torch.log(2 * torch.pi * self.var)
            - 1 / (2 * self.var) * (x - self.mean) ** 2
        )


def _brownian_dynamics(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    temperature: float,
    *,
    drag_coefficient: float = 1.0,
) -> Tuple[Callable[..., T], Callable[[T], T]]:
    force_fn = _canonicalize_force_fn(fn)

    # step_size, drag_coefficient = _static_cast(step_size, drag_coefficient)

    def _setup_fn(positions: Tensor, masses: float | Tensor = 1.0) -> T:
        if isinstance(masses, float):
            masses = torch.tensor(
                masses,
                device=positions.device,
                dtype=positions.dtype,
            )

        state = _BrownianDynamicsState(masses=masses, positions=positions)

        return _canonicalize_masses(state)

    def _step_fn(state: T, **kwargs) -> T:
        if "temperature" not in kwargs:
            _temperature = temperature
        else:
            _temperature = kwargs["temperature"]

        positions, masses = dataclasses.astuple(state)

        forces = force_fn(positions, **kwargs)

        xi = torch.randn(
            positions.shape,
            device=positions.device,
            dtype=positions.dtype,
        )

        nu = 1.0 / (masses * drag_coefficient)

        distances = (
            forces * step_size * nu
            + torch.sqrt(2.0 * _temperature * step_size * nu) * xi
        )

        positions = shift_fn(positions, distances, **kwargs)

        return _BrownianDynamicsState(masses=masses, positions=positions)

    return _setup_fn, _step_fn


def _nvt_langevin_thermostat(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    temperature: float,
    gamma: float = 0.1,
) -> Tuple[Callable[..., T], Callable[[T], T]]:
    force_fn = _canonicalize_force_fn(fn)

    def setup_fn(
        _positions: Tensor,
        _masses: float = 1.0,
        **kwargs,
    ) -> _NVTLangevinThermostatState:
        if isinstance(_masses, float):
            _masses = torch.tensor(_masses, dtype=_positions.dtype)

        forces = force_fn(_positions, **kwargs)

        state = _NVTLangevinThermostatState(forces, _masses, None, _positions)

        state = _canonicalize_masses(state)

        _temperature = kwargs.pop("temperature", temperature)

        return _setup_momentum(state, _temperature)

    def step_fn(
        state: _NVTLangevinThermostatState,
        **kwargs,
    ) -> _NVTLangevinThermostatState:
        _step_size = kwargs.pop("step_size", step_size)

        _temperature = kwargs.pop("temperature", temperature)

        state = _momentum_step(state, _step_size / 2)

        state = _positions_step(state, shift_fn, _step_size / 2, **kwargs)

        state = _stochastic_step(state, _step_size, _temperature, gamma)

        state = _positions_step(state, shift_fn, _step_size / 2, **kwargs)

        forces = force_fn(state.positions, **kwargs)

        state = state.set(forces=forces)

        return _momentum_step(state, _step_size / 2)

    return setup_fn, step_fn


def _canonicalize_distance_fn(
    fn: Callable[[Tensor, Tensor], Tensor],
) -> Callable[[Tensor, Tensor], Tensor]:
    """

    Parameters
    ----------
    fn

    Returns
    -------

    """
    for dimension in {1, 2, 3}:
        try:
            x = torch.rand([dimension], dtype=torch.float32)

            displacement = fn(x, x, t=0)

            if len(displacement.shape) == 0:
                return fn
            else:
                return _to_metric_fn(fn)
        except TypeError:
            continue
        except ValueError:
            continue

    raise ValueError


def _canonicalize_force_fn(fn: Callable[..., Tensor]) -> Callable[..., Tensor]:
    _force_fn = None

    def _fn(_positions: Tensor, **kwargs):
        nonlocal _force_fn

        if _force_fn is None:
            outputs = fn(_positions, **kwargs)

            if outputs.shape == ():
                _force_fn = force(fn)
            else:

                def _f(x: Tensor, y: Tensor) -> bool:
                    return x.shape == y.shape

                tree_map = optree.tree_map(_f, outputs, _positions)

                def _g(x, y):
                    return x and y

                if not optree.tree_reduce(_g, tree_map, True):
                    raise ValueError

                _force_fn = fn

        return _force_fn(_positions, **kwargs)

    return _fn


@_DispatchByState
def _canonicalize_masses(state: T) -> T:
    def _fn(_mass: float | Tensor) -> float | Tensor:
        if isinstance(_mass, float):
            return _mass

        match _mass.ndim:
            case 0:
                return _mass
            case 1:
                return torch.reshape(_mass, [_mass.shape[0], 1])
            case 2 if _mass.shape[1] == 1:
                return _mass

        raise ValueError

    masses = optree.tree_map(_fn, state.masses)

    return state.set(masses=masses)


def _cutoff(
    fn: Callable[..., Tensor],
    *,
    cutoff_radius: Tensor,
    turn_on_radius: Tensor,
) -> Callable[..., Tensor]:
    if isinstance(cutoff_radius, float):
        cutoff_radius = torch.tensor(cutoff_radius)

    if isinstance(turn_on_radius, float):
        turn_on_radius = torch.tensor(turn_on_radius)

    def _smooth_fn(distances: Tensor) -> Tensor:
        x = torch.square(cutoff_radius)
        y = torch.square(turn_on_radius)

        a = torch.square(x - torch.pow(distances, 2.0))

        b = x + torch.pow(distances, 2.0) * 2.0 - y * 3.0

        x = a * b / torch.pow(x - y, 3.0)

        return torch.where(
            torch.less(distances, turn_on_radius),
            1.0,
            torch.where(
                torch.less(distances, cutoff_radius),
                x,
                0.0,
            ),
        )

    @functools.wraps(fn)
    def _cutoff_fn(distances: Tensor, *args, **kwargs) -> Tensor:
        return _smooth_fn(distances) * fn(distances, *args, **kwargs)

    return _cutoff_fn


@functools.singledispatch
def _degrees_of_freedom_metric(positions: Tensor) -> int:
    # util.check_custom_simulation_type(position)

    def _fn(accumulator: Tensor, x: Tensor) -> int:
        return accumulator + torch.numel(x)

    return optree.tree_reduce(_fn, positions, 0)


def _fire_descent_optimizer(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    starting_step_size: float = 0.1,
    maximum_step_size: float = 0.4,
    minimum_steps: float = 5,
    positive_rate: float = 1.1,
    negative_rate: float = 0.5,
    initial_momentum: float = 0.1,
    change_in_momentum: float = 0.99,
) -> Tuple[Callable[..., T], Callable[[T], T]][_FIREDescentOptimizerState]:
    nve_setup_fn, nve_step_fn = _nve(fn, shift_fn, starting_step_size)

    force_fn = _canonicalize_force_fn(fn)

    def _setup_fn(
        positions: PyTree,
        masses: float | Tensor = 1.0,
        **kwargs,
    ) -> _FIREDescentOptimizerState:
        def _fn(x: Tensor):
            return torch.zeros_like(x)

        momentums = optree.tree_map(_fn, positions)

        n_pos = torch.zeros([], dtype=torch.int32)

        forces = force_fn(positions, **kwargs)

        state = _FIREDescentOptimizerState(
            current_momentum=initial_momentum,
            forces=forces,
            masses=masses,
            momentums=momentums,
            positions=positions,
            step_size=starting_step_size,
            steps=n_pos,
        )

        return _canonicalize_masses(state)

    def _step_fn(
        state: _FIREDescentOptimizerState, **kwargs
    ) -> _FIREDescentOptimizerState:
        state = nve_step_fn(state, step_size=state.step_size, **kwargs)

        (
            current_momentum,
            forces,
            masses,
            momentums,
            positions,
            step_size,
            steps,
        ) = _unpack(state)

        def _standardized_forces_fn(
            _accumulator: Tensor,
            _forces: Tensor,
        ) -> Tensor:
            if isinstance(_forces, float):
                _forces = torch.tensor(_forces)

            epsilon = torch.finfo(_forces.dtype).eps

            return _accumulator + torch.sum(torch.square(_forces)) + epsilon

        standardized_forces = optree.tree_reduce(
            _standardized_forces_fn,
            forces,
            torch.tensor(0.0),
        )

        standardized_forces = torch.sqrt(standardized_forces)

        def _normalized_momentums_fn(
            _accumulator: Tensor,
            _momentums: Tensor,
        ) -> Tensor:
            return _accumulator + torch.sum(_momentums**2)

        standardized_momentums = optree.tree_reduce(
            _normalized_momentums_fn,
            momentums,
            torch.tensor(0.0),
        )

        standardized_momentums = torch.sqrt(standardized_momentums)

        def _forces_momentums_product_fn(
            _forces: Tensor,
            _momentums: Tensor,
        ) -> Tensor:
            return torch.sum(_forces * _momentums)

        forces_momentums_product = optree.tree_map(
            _forces_momentums_product_fn,
            forces,
            momentums,
        )

        def _accumulate_forces_momentums_product_fn(
            _accumulator: Tensor,
            _forces_momentums_dot_product: Tensor,
        ) -> Tensor:
            return _accumulator + _forces_momentums_dot_product

        forces_momentums_product = optree.tree_reduce(
            _accumulate_forces_momentums_product_fn,
            forces_momentums_product,
        )

        def _momentum_fn(_momentums: Tensor, _forces: Tensor) -> Tensor:
            return _momentums + current_momentum * (
                _forces * standardized_momentums / standardized_forces
                - _momentums
            )

        momentums = optree.tree_map(_momentum_fn, momentums, forces)

        steps = torch.where(
            torch.greater_equal(forces_momentums_product, 0.0),
            steps + 1,
            0.0,
        )

        step_size_choice = [step_size * positive_rate, maximum_step_size]

        step_size_choice = torch.tensor(step_size_choice)

        step_size = torch.where(
            torch.greater(forces_momentums_product, 0.0),
            torch.where(
                torch.greater(steps, minimum_steps),
                torch.min(step_size_choice),
                step_size,
            ),
            step_size,
        )

        step_size = torch.where(
            torch.less(forces_momentums_product, 0.0),
            step_size * negative_rate,
            step_size,
        )

        current_momentum = torch.where(
            torch.greater(forces_momentums_product, 0.0),
            torch.where(
                torch.greater(steps, minimum_steps),
                current_momentum * change_in_momentum,
                current_momentum,
            ),
            current_momentum,
        )

        current_momentum = torch.where(
            torch.less(forces_momentums_product, 0.0),
            initial_momentum,
            current_momentum,
        )

        def _momentum_fn(_momentums: Tensor) -> Tensor:
            return (forces_momentums_product >= 0) * _momentums

        momentums = optree.tree_map(_momentum_fn, momentums)

        return _FIREDescentOptimizerState(
            current_momentum=current_momentum,
            forces=forces,
            masses=masses,
            momentums=momentums,
            positions=positions,
            step_size=step_size,
            steps=steps,
        )

    return _setup_fn, _step_fn


def _inverse_transform(transformation: Tensor) -> Tensor:
    """
    Calculates the inverse of an affine transformation matrix.

    Parameters
    ----------
    transformation : Tensor
        The affine transformation matrix to be inverted.

    Returns
    -------
    Tensor
        The inverse of the given affine transformation matrix.
    """
    if transformation.ndim in {0, 1}:
        return 1.0 / transformation

    if transformation.ndim == 2:
        return torch.linalg.inv(transformation)

    raise ValueError("Unsupported transformation dimensions.")


@_DispatchByState
def _update_kinetic_energy(state: T) -> Tensor:
    return _kinetic_energy_metric(
        masses=state.masses, momentums=state.momentums
    )


def _kinetic_energy_metric(
    *,
    momentums: Tensor = None,
    velocities: Tensor = None,
    masses: Tensor = 1.0,
) -> Tensor:
    if momentums is not None and velocities is not None:
        raise ValueError

    if momentums is not None:
        momentums_or_velocities = momentums
    else:
        momentums_or_velocities = velocities

    # _check_custom_simulation_type(q)

    def _kinetic_energy_fn(
        _masses: Tensor,
        _momentums_or_velocities: Tensor,
    ) -> Tensor:
        if momentums is None:

            def k(v, m):
                return v**2 * m
        else:

            def k(p, m):
                return p**2 / m

        return 0.5 * _safe_sum(k(_momentums_or_velocities, _masses))

    kinetic_energy = optree.tree_map(
        _kinetic_energy_fn,
        masses,
        momentums_or_velocities,
    )

    return optree.tree_reduce(operator.add, kinetic_energy, 0.0)


def _map_product(
    fn: Callable[[Tensor, Tensor], Tensor],
) -> Callable[[Tensor, Tensor], Tensor]:
    """

    Parameters
    ----------
    fn

    Returns
    -------

    """
    return torch.func.vmap(
        torch.func.vmap(
            fn,
            (0, None),
            0,
        ),
        (None, 0),
        0,
    )


@_DispatchByState
def _momentum_step(state: T, step_size: float) -> T:
    def _fn(_momentums: Tensor, _forces: Tensor) -> Tensor:
        return _momentums + step_size * _forces

    momentums = optree.tree_map(_fn, state.momentums, state.forces)

    return state.set(momentums=momentums)


def _nve(fn, shift_fn, step_size: float = 0.001, **_):
    force_fn = _canonicalize_force_fn(fn)

    def _setup_fn(_positions, _temperature, _masses=1.0, **_kwargs):
        if not isinstance(_temperature, Tensor):
            _temperature = torch.tensor(_temperature, dtype=_positions.dtype)

        state = _NVEState(
            forces=force_fn(_positions, **_kwargs),
            masses=_masses,
            momentums=None,
            positions=_positions,
        )

        state = _canonicalize_masses(state)

        return _setup_momentum(state, _temperature)

    def step_fn(_state, **kwargs):
        _step_size = kwargs.pop("step_size", step_size)

        return _velocity_verlet(
            force_fn,
            shift_fn,
            _step_size,
            _state,
            **kwargs,
        )

    return _setup_fn, step_fn


def _particle_density_to_size(
    particles: int,
    density: float,
    spatial_dimension: int,
) -> float:
    return math.pow(particles / density, 1 / spatial_dimension)


@_DispatchByState
def _positions_step(
    state: T,
    shift_fn: Callable,
    step_size: float,
    **kwargs,
) -> T:
    if isinstance(shift_fn, Callable):

        def _fn(_: Tensor) -> Callable:
            return shift_fn

        shift_fn = optree.tree_map(_fn, state.positions)

    def _fn(
        _shift_fn: Callable,
        _positions: Tensor,
        _momentums: Tensor,
        _masses: Tensor,
    ) -> Tensor:
        return _shift_fn(
            _positions,
            step_size * _momentums / _masses,
            **kwargs,
        )

    positions = optree.tree_map(
        _fn,
        shift_fn,
        state.positions,
        state.momentums,
        state.masses,
    )

    return state.set(positions=positions)


def _safe_mask(mask, fn, operand, placeholder: float = 0.0):
    return torch.where(mask, fn(torch.where(mask, operand, 0)), placeholder)


def _safe_sum(
    x: Tensor,
    dim: Optional[Union[Iterable[int], int]] = None,
    keepdim: bool = False,
):
    match x:
        case _ if x.is_complex():
            promoted_dtype = torch.complex128
        case _ if x.is_floating_point():
            promoted_dtype = torch.float64
        case _:
            promoted_dtype = torch.int64

    summation = torch.sum(x, dim=dim, dtype=promoted_dtype, keepdim=keepdim)

    return summation.to(dtype=x.dtype)


@_DispatchByState
def _setup_momentum(state: T, temperature: float) -> T:
    positions, masses = state.positions, state.masses

    positions, tree_spec = optree.tree_flatten(positions)

    masses, _ = optree.tree_flatten(masses)

    def _fn(_position: Tensor, _mass: Tensor) -> Tensor:
        sample = torch.normal(
            0.0,
            1.0,
            _position.shape,
            device=_position.device,
            dtype=_position.dtype,
        )

        momentum = torch.sqrt(_mass * temperature) * sample

        if _position.shape[0] > 1:
            momentum = momentum - torch.mean(momentum, dim=0, keepdim=True)

        return momentum

    momentums = []

    for position, mass in zip(positions, masses):
        momentums = [*momentums, _fn(position, mass)]

    momentums = optree.tree_unflatten(tree_spec, momentums)

    return state.set(momentums=momentums)


@_DispatchByState
def _stochastic_step(
    state: _NVTLangevinThermostatState,
    step_size: float,
    temperature: float,
    gamma: float,
) -> _NVTLangevinThermostatState:
    c1 = math.exp(-gamma * step_size)

    c2 = math.sqrt(temperature * (1 - c1**2))

    momentum_dist = Normal(c1 * state.momentums, c2**2 * state.masses)

    return state.set(momentums=momentum_dist.sample())


@_DispatchByState
def _update_temperature(state: T) -> Tensor:
    return _temperature_metric(masses=state.masses, momentums=state.momentums)


def _temperature_metric(
    *,
    momentums: Tensor = None,
    velocities: Tensor = None,
    masses: Tensor = 1.0,
) -> Tensor:
    if momentums is not None and velocities is not None:
        raise ValueError

    if momentums is None:

        def t(v, m):
            return v**2 * m
    else:

        def t(p, m):
            return p**2 / m

    if momentums is None:
        q = velocities
    else:
        q = momentums

    # _check_custom_simulation_type(q)

    def m_dof(m, q):
        return _safe_sum(t(q, m)) / _degrees_of_freedom_metric(q)

    temperature = optree.tree_map(m_dof, masses, q)

    return optree.tree_reduce(operator.add, temperature, 0.0)


def _to_metric_fn(
    fn: Callable[[Tensor, Tensor], Tensor],
) -> Callable[[Tensor, Tensor], Tensor]:
    """

    Parameters
    ----------
    fn

    Returns
    -------

    """

    def _fn(a: Tensor, b: Tensor, **kwargs):
        input = torch.sum(
            torch.square(fn(a, b, **kwargs)),
            dim=-1,
        )

        return torch.where(
            torch.greater(input, 0.0),
            torch.sqrt(torch.where(torch.greater(input, 0.0), input, 0.0)),
            0.0,
        )

    return _fn


def _transform(transformation: Tensor, position: Tensor) -> Tensor:
    """
    Applies an affine transformation to the position vector.

    Parameters
    ----------
    position : Tensor
        Position, must have the shape `(..., dimension)`.

    transformation : Tensor
        The affine transformation matrix, must be a scalar, a vector, or a
        matrix with the shape `(dimension, dimension)`.

    Returns
    -------
    Tensor
        Affine transformed position vector, has the same shape as the
        position vector.
    """
    if transformation.ndim == 0:
        return position * transformation

    indices = [chr(ord("a") + index) for index in range(position.ndim - 1)]

    indices = "".join(indices)

    if transformation.ndim == 1:
        return torch.einsum(
            f"i,{indices}i->{indices}i",
            transformation,
            position,
        )

    if transformation.ndim == 2:
        return torch.einsum(
            f"ij,{indices}j->{indices}i",
            transformation,
            position,
        )

    raise ValueError("Unsupported transformation dimensions.")


def transform(transformation: Tensor, position: Tensor) -> Tensor:
    """
    Return affine transformed position.

    Parameters
    ----------
    transformation : Tensor
        Affine transformation matrix, must have shape
        `(dimension, dimension)`.

    position : Tensor
        Position, must have shape `(..., dimension)`.

    Returns
    -------
    Tensor
        Affine transformed position of shape `(..., dimension)`.
    """

    class _Transform(Function):
        @staticmethod
        def forward(transformation: Tensor, position: Tensor) -> Tensor:
            """
            Return affine transformed position.

            Parameters
            ----------
            transformation : Tensor
                Affine transformation matrix, must have shape
                `(dimension, dimension)`.

            position : Tensor
                Position, must have shape `(..., dimension)`.

            Returns
            -------
            Tensor
                Affine transformed position of shape `(..., dimension)`.
            """
            return _transform(transformation, position)

        @staticmethod
        def setup_context(ctx, inputs, output):
            transformation, position = inputs

            ctx.save_for_backward(transformation, position, output)

        @staticmethod
        def jvp(
            ctx, grad_transformation: Tensor, grad_position: Tensor
        ) -> Tuple[Tensor, Tensor]:
            transformation, position, _ = ctx.saved_tensors

            output = _transform(transformation, position)

            grad_output = grad_position + _transform(
                grad_transformation,
                position,
            )

            return output, grad_output

        @staticmethod
        def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
            _, _, output = ctx.saved_tensors

            return output, grad_output

    return _Transform.apply(transformation, position)


def _unpack(dc) -> tuple:
    return tuple(getattr(dc, field.name) for field in dataclasses.fields(dc))


def _velocity_verlet(
    force_fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    state: T,
    **kwargs,
) -> T:
    state = _momentum_step(state, step_size / 2)

    state = _positions_step(state, shift_fn, step_size, **kwargs)

    state = state.set(forces=force_fn(state.positions, **kwargs))

    return _momentum_step(state, step_size / 2)


def _volume_metric(dimension: int, box: Tensor) -> Tensor:
    if torch.tensor(box).shape == torch.Size([]) or not box.ndim:
        return box**dimension

    match box.ndim:
        case 1:
            return torch.prod(box)
        case 2:
            return torch.linalg.det(box)
        case _:
            raise ValueError


def _zero_diagonal_mask(x: Tensor) -> Tensor:
    """Sets the diagonal of a matrix to zero."""
    if x.shape[0] != x.shape[1]:
        raise ValueError(
            f"Diagonal mask can only mask square matrices. Found {x.shape[0]}x{x.shape[1]}."
        )

    if len(x.shape) > 3:
        raise ValueError(
            f"Diagonal mask can only mask rank-2 or rank-3 tensors. Found {len(x.shape)}."
        )

    n = x.shape[0]

    x = torch.nan_to_num(x)

    mask = 1.0 - torch.eye(n, device=x.device, dtype=x.dtype)

    if len(x.shape) == 3:
        mask = torch.reshape(mask, [n, n, 1])

    return x * mask


def distance(displacements: Tensor) -> Tensor:
    dr = torch.sum(torch.square(displacements), dim=-1)

    return _safe_mask(dr > 0.0, torch.sqrt, dr)


def simulation(
    fn: Callable[..., Tensor],
    shift_fn: Callable[[Tensor, Tensor], Tensor],
    step_size: float,
    temperature: float,
    gamma: float = 0.1,
) -> Tuple[Callable[..., T], Callable[[T], T]]:
    return _nvt_langevin_thermostat(
        fn,
        shift_fn,
        step_size,
        temperature,
        gamma,
    )


def space(
    dimensions: Optional[Tensor] = None,
    *,
    normalized: bool = True,
    parallelepiped: bool = True,
    remapped: bool = True,
) -> Tuple[Callable, Callable]:
    r"""Define a simulation space.

    This function is fundamental in constructing simulation spaces derived from
    subsets of $\mathbb{R}^{D}$ (where $D = 1$, $2$, or $3$) and is
    instrumental in setting up simulation environments with specific
    characteristics (e.g., periodic boundary conditions). The function returns
    a a displacement function and a shift function to compute particle
    interactions and movements in space.

    This function supports deformation of the simulation cell, crucial for
    certain types of simulations, such as those involving finite deformations
    or the computation of elastic constants.

    Parameters
    ----------
    dimensions : Optional[Tensor], default=None
        Dimensions of the simulation space. Interpretation varies based on the
        value of `parallelepiped`. If `parallelepiped` is `True`, must be an
        affine transformation, $T$, specified in one of three ways: a cube,
        $L$; an orthorhombic unit cell, $[L_{x}, L_{y}, L_{z}]$; or a triclinic
        cell, upper triangular matrix. If `parallelepiped` is `False`, must be
        the edge lengths. If `None`, the simulation space has free boudnary
        conditions.

    normalized : bool, default=True
        If `True`, positions are stored in the unit cube. Displacements and
        shifts are computed in a normalized simulation space and can be
        transformed back to real simulation space using the provided affine
        transformation matrix. If `False`, positions are expressed and
        computations performed directly in the real simulation space.

    parallelepiped : bool, default=True
        If `True`, the simulation space is defined as a ${1, 2, 3}$-dimensional
        parallelepiped with periodic boundary conditions. If `False`, the space
        is defined on a ${1, 2, 3}$-dimensional hypercube.

    remapped : bool, default=True
        If `True`, positions and displacements are remapped to stay in the
        bounds of the defined simulation space. A rempapped simulation space is
        topologically equivalent to a torus, ensuring that particles exiting
        one boundary re-enter from the opposite side. This is particularly
        relevant for simulation spaces with periodic boundary conditions.

    Returns
    -------
    Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor, Tensor], Tensor]]
        A tuple containing two functions:

        1.  The displacement function, $\overrightarrow{d}$, measures the
            difference between two points in the simulation space, factoring in
            the geometry and boundary conditions. This function is used to
            calculate particle interactions and dynamics.
        2.  The shift function, $u$, applies a displacement vector to a point
            in the space, effectively moving it. This function is used to
            update simulated particle positions.

    Examples
    --------
        transformation = torch.tensor([10.0])

        displacement_fn, shift_fn = space(
            transformation,
            normalized=False,
        )

        normalized_displacement_fn, normalized_shift_fn = space(
            transformation,
            normalized=True,
        )

        normalized_position = torch.rand([4, 3])

        position = transformation * normalized_position

        displacement = torch.randn([4, 3])

        torch.testing.assert_close(
            displacement_fn(position[0], position[1]),
            normalized_displacement_fn(
                normalized_position[0],
                normalized_position[1],
            ),
        )
    """
    if dimensions is None:

        def _displacement_fn(
            a: Tensor,
            b: Tensor,
            *,
            perturbation: Optional[Tensor] = None,
            **_,
        ) -> Tensor:
            if len(a.shape) != 1:
                raise ValueError

            if a.shape != b.shape:
                raise ValueError

            if perturbation is not None:
                return _transform(a - b, perturbation)

            return a - b

        def _shift_fn(a: Tensor, b: Tensor, **_) -> Tensor:
            return a + b

        return _displacement_fn, _shift_fn

    if parallelepiped:
        inverse_transformation = _inverse_transform(dimensions)

        if normalized:

            def _displacement_fn(
                a: Tensor,
                b: Tensor,
                *,
                perturbation: Optional[Tensor] = None,
                **kwargs,
            ) -> Tensor:
                _transformation = dimensions

                _inverse_transformation = inverse_transformation

                if "transformation" in kwargs:
                    _transformation = kwargs["transformation"]

                if "updated_transformation" in kwargs:
                    _transformation = kwargs["updated_transformation"]

                if len(a.shape) != 1:
                    raise ValueError

                if a.shape != b.shape:
                    raise ValueError

                displacement = a - b

                displacement = torch.remainder(displacement + 1.0 * 0.5, 1.0)

                displacement = displacement - 1.0 * 0.5

                displacement = transform(_transformation, displacement)

                if perturbation is not None:
                    return _transform(displacement, perturbation)

                return displacement

            if remapped:

                def _u(a: Tensor, b: Tensor) -> Tensor:
                    return torch.remainder(a + b, 1.0)

                def _shift_fn(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                    _transformation = dimensions

                    _inverse_transformation = inverse_transformation

                    if "transformation" in kwargs:
                        _transformation = kwargs["transformation"]

                        _inverse_transformation = _inverse_transform(
                            _transformation
                        )

                    if "updated_transformation" in kwargs:
                        _transformation = kwargs["updated_transformation"]

                    return _u(a, transform(_inverse_transformation, b))

                return _displacement_fn, _shift_fn

            def _shift_fn(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                _transformation = dimensions

                _inverse_transformation = inverse_transformation

                if "transformation" in kwargs:
                    _transformation = kwargs["transformation"]

                    _inverse_transformation = _inverse_transform(
                        _transformation,
                    )

                if "updated_transformation" in kwargs:
                    _transformation = kwargs["updated_transformation"]

                return a + transform(_inverse_transformation, b)

            return _displacement_fn, _shift_fn

        def _displacement_fn(
            a: Tensor,
            b: Tensor,
            *,
            perturbation: Optional[Tensor] = None,
            **kwargs,
        ) -> Tensor:
            _transformation = dimensions

            _inverse_transformation = inverse_transformation

            if "transformation" in kwargs:
                _transformation = kwargs["transformation"]

                _inverse_transformation = _inverse_transform(_transformation)

            if "updated_transformation" in kwargs:
                _transformation = kwargs["updated_transformation"]

            a = transform(_inverse_transformation, a)
            b = transform(_inverse_transformation, b)

            if len(a.shape) != 1:
                raise ValueError

            if a.shape != b.shape:
                raise ValueError

            displacement = a - b

            displacement = torch.remainder(displacement + 1.0 * 0.5, 1.0)

            displacement = displacement - 1.0 * 0.5

            displacement = transform(_transformation, displacement)

            if perturbation is not None:
                return _transform(displacement, perturbation)

            return displacement

        if remapped:

            def _u(a: Tensor, b: Tensor) -> Tensor:
                return torch.remainder(a + b, 1.0)

            def _shift_fn(a: Tensor, b: Tensor, **kwargs) -> Tensor:
                _transformation = dimensions

                _inverse_transformation = inverse_transformation

                if "transformation" in kwargs:
                    _transformation = kwargs["transformation"]

                    _inverse_transformation = _inverse_transform(
                        _transformation,
                    )

                if "updated_transformation" in kwargs:
                    _transformation = kwargs["updated_transformation"]

                return transform(
                    _transformation,
                    _u(
                        transform(_inverse_transformation, a),
                        transform(_inverse_transformation, b),
                    ),
                )

            return _displacement_fn, _shift_fn

        def _shift_fn(a: Tensor, b: Tensor, **_) -> Tensor:
            return a + b

        return _displacement_fn, _shift_fn

    def _displacement_fn(
        a: Tensor,
        b: Tensor,
        *,
        perturbation: Optional[Tensor] = None,
        **_,
    ) -> Tensor:
        if len(a.shape) != 1:
            raise ValueError

        if a.shape != b.shape:
            raise ValueError

        displacement = torch.remainder(a - b + dimensions * 0.5, dimensions)

        displacement = displacement - dimensions * 0.5

        if perturbation is not None:
            return _transform(displacement, perturbation)

        return displacement

    if remapped:

        def _shift_fn(a: Tensor, b: Tensor, **_) -> Tensor:
            return torch.remainder(a + b, dimensions)
    else:

        def _shift_fn(a: Tensor, b: Tensor, **_) -> Tensor:
            return a + b

    return _displacement_fn, _shift_fn
