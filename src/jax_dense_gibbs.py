"""Exact complex128 statevector objectives for small Gibbs benchmarks.

This backend mirrors the Qiskit circuits produced by ``BasicAnsatz`` and
``TFDAnsatz`` without converting them to an MPS. It is intended for the
small ``N=4,6`` validation benchmark, where a ``2**(2*N)`` statevector and a
``2**N`` reduced density matrix are inexpensive.  Qubit zero is the least
significant statevector bit, matching Qiskit's statevector convention.

Entropy is measured in nats.  The regularized analytic spectral VJP used by
the production MPS objective is reused so rank-deficient HEA density matrices
do not differentiate through generic eigenvector sensitivities.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Integral, Real
from typing import ClassVar, NamedTuple, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

try:
    from .jax_mps_gibbs import _entropy_nats_from_density_matrix
except ImportError:  # Support the repository's historical top-level imports.
    from jax_mps_gibbs import _entropy_nats_from_density_matrix


Array = jax.Array


class ThermodynamicValues(NamedTuple):
    """JAX scalar totals F and E, entropy in nats, norm, and minimum weight.

    The last field is the smallest normalized, regularized density-matrix
    weight and can be zero after spectral truncation.
    """

    free_energy: Array
    energy: Array
    entropy_nats: Array
    norm: Array
    minimum_density_weight: Array


def _integer(name: str, value: object, minimum: int) -> int:
    """Return an integer at least ``minimum``, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _real(name: str, value: object) -> float:
    """Return a finite real scalar as float, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DenseGibbsObjectiveSpec:
    """Frozen exact-statevector objective for TFDA or contiguous HEA.

    ``model`` is ``"tfim"`` or ``"xxz"`` and ``ansatz_type`` is ``"tfda"``
    or ``"hea"``.  Ancillas occupy Qiskit qubits ``0..N_a-1`` and the system
    occupies the remaining higher-index qubits.  The benchmark requires
    ``N_a=N`` for both supported ansatzes.
    """

    optimizer_backend_module: ClassVar[str] = "jax_dense_gibbs"

    name: str
    model: str
    ansatz_type: str
    num_system: int
    num_ancillas: int
    num_layers: int
    beta: float
    edges: tuple[tuple[int, int], ...]
    tfim_j: float = -1.0
    tfim_hz: float = 0.0
    tfim_hx: float = -0.5
    xxz_jxy: float = -1.0
    xxz_jz: float = 1.5

    def __post_init__(self) -> None:
        """Validate benchmark dimensions, beta, couplings, and undirected edges.

        Model names are normalized to lowercase and each edge is sorted;
        duplicate edges, self-edges, and unequal system/ancilla sizes fail.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        model = str(self.model).strip().lower()
        ansatz_type = str(self.ansatz_type).strip().lower()
        if model not in {"tfim", "xxz"}:
            raise ValueError("model must be 'tfim' or 'xxz'")
        if ansatz_type not in {"tfda", "hea"}:
            raise ValueError("ansatz_type must be 'tfda' or 'hea'")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "ansatz_type", ansatz_type)

        num_system = _integer("num_system", self.num_system, 1)
        num_ancillas = _integer("num_ancillas", self.num_ancillas, 1)
        num_layers = _integer("num_layers", self.num_layers, 1)
        if num_ancillas != num_system:
            raise ValueError("the small benchmark requires num_ancillas == num_system")
        object.__setattr__(self, "num_system", num_system)
        object.__setattr__(self, "num_ancillas", num_ancillas)
        object.__setattr__(self, "num_layers", num_layers)

        beta = _real("beta", self.beta)
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        object.__setattr__(self, "beta", beta)
        for field in ("tfim_j", "tfim_hz", "tfim_hx", "xxz_jxy", "xxz_jz"):
            object.__setattr__(self, field, _real(field, getattr(self, field)))

        canonical: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for raw_edge in tuple(self.edges):
            if len(raw_edge) != 2:
                raise ValueError("each edge must contain two sites")
            first = _integer("edge site", raw_edge[0], 0)
            second = _integer("edge site", raw_edge[1], 0)
            if first == second or first >= num_system or second >= num_system:
                raise ValueError(f"invalid edge {raw_edge!r}")
            edge = (min(first, second), max(first, second))
            if edge in seen:
                raise ValueError(f"duplicate edge {edge}")
            seen.add(edge)
            canonical.append(edge)
        object.__setattr__(self, "edges", tuple(canonical))

    @property
    def num_qubits(self) -> int:
        """Return the total number of system and ancilla qubits."""
        return self.num_system + self.num_ancillas

    @property
    def num_parameters(self) -> int:
        """Return four angles per TFDA layer or one per HEA qubit and layer."""
        if self.ansatz_type == "tfda":
            return 4 * self.num_layers
        return self.num_qubits * self.num_layers


def _pauli_action_arrays(
    num_qubits: int, qubits: Sequence[int], labels: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays implementing ``P @ state = phase * state[source]``.

    Qubit zero is the least-significant bit; each array has ``2**num_qubits``
    entries. Labels are X, Y, or Z for the corresponding qubit indices.
    """

    dimension = 1 << num_qubits
    indices = np.arange(dimension, dtype=np.uint32)
    source = indices.copy()
    phase = np.ones(dimension, dtype=np.complex128)
    for qubit, label in zip(qubits, labels, strict=True):
        bit = (indices >> np.uint32(qubit)) & np.uint32(1)
        if label == "X":
            source ^= np.uint32(1 << qubit)
        elif label == "Y":
            source ^= np.uint32(1 << qubit)
            phase *= 1j * (2.0 * bit.astype(np.float64) - 1.0)
        elif label == "Z":
            phase *= 1.0 - 2.0 * bit.astype(np.float64)
        else:
            raise ValueError(f"unsupported Pauli label {label!r}")
    return source.astype(np.int32), phase


def _cnot_action_arrays(
    num_qubits: int, control: int, target: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the little-endian CNOT gather permutation and unit phases."""
    dimension = 1 << num_qubits
    indices = np.arange(dimension, dtype=np.uint32)
    control_bit = (indices >> np.uint32(control)) & np.uint32(1)
    source = indices ^ (control_bit << np.uint32(target))
    return source.astype(np.int32), np.ones(dimension, dtype=np.complex128)


@lru_cache(maxsize=32)
def _initial_state_and_operations(
    spec: DenseGibbsObjectiveSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode and cache circuit operations as compact JAX scan inputs.

    Returns:
        Initial state, source indices, phases, parameter indices, and rotation
        flags. The two action arrays have shape ``(gates, 2**num_qubits)``;
        parameter indices and flags each have one entry per gate. TFDA starts
        from Bell pairs and HEA starts from the all-zero state.
    """

    dimension = 1 << spec.num_qubits
    initial = np.zeros(dimension, dtype=np.complex128)
    sources: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    parameter_indices: list[int] = []
    rotation_flags: list[bool] = []

    def rotation(
        parameter_index: int, qubits: Sequence[int], labels: Sequence[str]
    ) -> None:
        """Append a Pauli rotation with a shared parameter index to the scan."""
        source, phase = _pauli_action_arrays(spec.num_qubits, qubits, labels)
        sources.append(source)
        phases.append(phase)
        parameter_indices.append(parameter_index)
        rotation_flags.append(True)

    def cnot(control: int, target: int) -> None:
        """Append a parameter-independent CNOT to the operation lists."""
        source, phase = _cnot_action_arrays(spec.num_qubits, control, target)
        sources.append(source)
        phases.append(phase)
        # A safe in-bounds placeholder; the flag selects the fixed operation.
        parameter_indices.append(0)
        rotation_flags.append(False)

    if spec.ansatz_type == "hea":
        initial[0] = 1.0
        for layer in range(spec.num_layers):
            offset = layer * spec.num_qubits
            for qubit in range(spec.num_qubits):
                rotation(offset + qubit, (qubit,), ("Y",))
            for qubit in range(spec.num_qubits - 1):
                cnot(qubit, qubit + 1)
    else:
        n = spec.num_system
        configurations = np.arange(1 << n, dtype=np.uint32)
        bell_indices = configurations | (configurations << np.uint32(n))
        initial[bell_indices] = 1.0 / np.sqrt(float(1 << n))
        for layer in range(spec.num_layers):
            theta_0 = 4 * layer
            theta_1 = theta_0 + 1
            theta_2 = theta_0 + 2
            theta_3 = theta_0 + 3
            for first, second in spec.edges:
                if spec.model == "tfim":
                    rotation(theta_0, (first, second), ("Z", "Z"))
                    rotation(theta_0, (first + n, second + n), ("Z", "Z"))
                else:
                    rotation(theta_0, (first, second), ("X", "X"))
                    rotation(theta_0, (first + n, second + n), ("X", "X"))
                    rotation(theta_0, (first, second), ("Y", "Y"))
                    rotation(theta_0, (first + n, second + n), ("Y", "Y"))
                    rotation(theta_1, (first, second), ("Z", "Z"))
                    rotation(theta_1, (first + n, second + n), ("Z", "Z"))
            for site in range(n):
                if spec.model == "tfim":
                    rotation(theta_1, (site,), ("X",))
                    rotation(theta_1, (site + n,), ("X",))
                rotation(theta_2, (site, site + n), ("X", "X"))
                rotation(theta_3, (site, site + n), ("Z", "Z"))

    return (
        initial,
        np.stack(sources, axis=0),
        np.stack(phases, axis=0),
        np.asarray(parameter_indices, dtype=np.int32),
        np.asarray(rotation_flags, dtype=np.bool_),
    )


def statevector(params: Array, spec: DenseGibbsObjectiveSpec) -> Array:
    """Construct the exact variational statevector with one compact scan.

    Args:
        params: Flat angle vector in radians, with four shared parameters per
            TFDA layer or layer-major qubit rotations for HEA.
        spec: Static circuit specification in little-endian qubit order.

    Returns:
        Complex128 vector with ``2**spec.num_qubits`` amplitudes.
    """

    params = jnp.asarray(params, dtype=jnp.float64)
    initial, sources, phases, parameter_indices, rotation_flags = (
        _initial_state_and_operations(spec)
    )

    def apply_operation(state: Array, operation):
        """Apply a fixed gate or ``exp(-i*theta*P/2)`` as one scan step."""
        source, phase, parameter_index, is_rotation = operation
        transformed = phase * state[source]
        theta = params[parameter_index]
        rotated = (
            jnp.cos(theta / 2.0) * state
            - 1j * jnp.sin(theta / 2.0) * transformed
        )
        next_state = jnp.where(is_rotation, rotated, transformed)
        return next_state, None

    state, _ = jax.lax.scan(
        apply_operation,
        jnp.asarray(initial, dtype=jnp.complex128),
        (
            jnp.asarray(sources),
            jnp.asarray(phases),
            jnp.asarray(parameter_indices),
            jnp.asarray(rotation_flags),
        ),
    )
    return state


def reduced_system_density_matrix(
    params: Array, spec: DenseGibbsObjectiveSpec
) -> tuple[Array, Array]:
    """Trace out the low-index ancillas and normalize the system state.

    Returns:
        Hermitian ``(2**N, 2**N)`` system density matrix and the squared norm
        of the unnormalized purification statevector, where N is system size.
    """

    state = statevector(params, spec)
    norm = jnp.real(jnp.vdot(state, state))
    amplitudes = jnp.reshape(
        state, (1 << spec.num_system, 1 << spec.num_ancillas)
    )
    rho = amplitudes @ jnp.conj(amplitudes.T)
    rho = (rho + jnp.conj(rho.T)) / (2.0 * norm)
    return rho, norm


def _pauli_matrix(
    num_qubits: int, qubits: Sequence[int], labels: Sequence[str]
) -> np.ndarray:
    """Build a dense Pauli-product matrix using little-endian qubit indices."""
    dimension = 1 << num_qubits
    indices = np.arange(dimension, dtype=np.uint64)
    source = indices.copy()
    phase = np.ones(dimension, dtype=np.complex128)
    for qubit, label in zip(qubits, labels, strict=True):
        bit = (indices >> np.uint64(qubit)) & np.uint64(1)
        if label == "X":
            source ^= np.uint64(1 << qubit)
        elif label == "Y":
            source ^= np.uint64(1 << qubit)
            phase *= 1j * (2.0 * bit.astype(np.float64) - 1.0)
        elif label == "Z":
            phase *= 1.0 - 2.0 * bit.astype(np.float64)
        else:
            raise ValueError(f"unsupported Pauli label {label!r}")
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    matrix[indices, source] = phase
    return matrix


def hamiltonian_matrix(spec: DenseGibbsObjectiveSpec) -> np.ndarray:
    """Construct the ``2**N``-square Hamiltonian in little-endian site order.

    TFIM uses signed coefficients for ZZ edges and onsite Z/X terms. XXZ uses
    ``jxy*(XX + YY) + jz*ZZ`` on each edge. These are Pauli, not spin-1/2,
    operators, and the matrix acts only on the N logical system qubits.
    """

    dimension = 1 << spec.num_system
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    if spec.model == "tfim":
        for first, second in spec.edges:
            matrix += spec.tfim_j * _pauli_matrix(
                spec.num_system, (first, second), ("Z", "Z")
            )
        for site in range(spec.num_system):
            if spec.tfim_hz:
                matrix += spec.tfim_hz * _pauli_matrix(
                    spec.num_system, (site,), ("Z",)
                )
            if spec.tfim_hx:
                matrix += spec.tfim_hx * _pauli_matrix(
                    spec.num_system, (site,), ("X",)
                )
    else:
        for first, second in spec.edges:
            matrix += spec.xxz_jxy * _pauli_matrix(
                spec.num_system, (first, second), ("X", "X")
            )
            matrix += spec.xxz_jxy * _pauli_matrix(
                spec.num_system, (first, second), ("Y", "Y")
            )
            matrix += spec.xxz_jz * _pauli_matrix(
                spec.num_system, (first, second), ("Z", "Z")
            )
    return matrix


def thermodynamics(params: Array, spec: DenseGibbsObjectiveSpec) -> ThermodynamicValues:
    """Evaluate total energy, regularized entropy, and ``F = E - S/beta``.

    Entropy uses natural logs and the shared masked, renormalized spectral
    helper. The returned norm is squared statevector norm; the optimizer
    separately rescales the total F to ``C = beta*F/N``.
    """

    if not isinstance(spec, DenseGibbsObjectiveSpec):
        raise TypeError("spec must be a DenseGibbsObjectiveSpec")
    rho, norm = reduced_system_density_matrix(params, spec)
    hamiltonian = jnp.asarray(hamiltonian_matrix(spec), dtype=jnp.complex128)
    energy = jnp.real(jnp.einsum("ij,ji->", hamiltonian, rho, optimize=True))
    entropy, minimum_weight = _entropy_nats_from_density_matrix(rho)
    free_energy = energy - entropy / spec.beta
    return ThermodynamicValues(free_energy, energy, entropy, norm, minimum_weight)


def make_loss_with_aux(spec: DenseGibbsObjectiveSpec):
    """Return a total-F loss closure with ``(E, S, norm, min_weight)`` aux.

    Entropy is in nats, and the captured specification remains static.
    """

    if not isinstance(spec, DenseGibbsObjectiveSpec):
        raise TypeError("spec must be a DenseGibbsObjectiveSpec")

    def loss_with_aux(params: Array):
        """Evaluate total F and the four physical diagnostics for one vector."""
        values = thermodynamics(params, spec)
        return values.free_energy, (
            values.energy,
            values.entropy_nats,
            values.norm,
            values.minimum_density_weight,
        )

    return loss_with_aux


__all__ = [
    "DenseGibbsObjectiveSpec",
    "ThermodynamicValues",
    "hamiltonian_matrix",
    "make_loss_with_aux",
    "reduced_system_density_matrix",
    "statevector",
    "thermodynamics",
]
