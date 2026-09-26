r"""Float64 JAX objectives for exact-MPS Gibbs-state optimization.

The original supported circuit keeps all ancillas before all system qubits and
uses a nearest-neighbour CNOT ladder.  The optional ``interleaved_pair`` layout
inserts each ancilla immediately before its paired system site and separates
pair CNOTs from the logical-system CNOT staircase.  Range-two system CNOTs are
applied as exact rank-two operator bridges across the intervening ancilla;
there are no physical SWAPs or truncations.  The conservative interleaved bond
ceiling is ``4**layers``.

Hamiltonian coefficients in ``JAXMPSObjectiveSpec`` are signed coefficients
in ``H = j*sum_edges(ZZ) + hz*sum_sites(Z) + hx*sum_sites(X)``.

The entropy is measured in nats and the objective is ``F = E - S / beta``.
All MPS tensors use float64 with axes ``(left bond, physical site, right
bond)``. Energy transfers assume real-valued tensors, as produced by the
supported RY/CNOT circuits; they do not conjugate a complex bra.

Energy evaluation caches the identity-transfer environment to the left and
right of every tensor.  A local term then contracts only its support and, for
a non-local edge in the MPS ordering, the short identity bridge between its
endpoints.  This is mathematically equivalent to a sparse Hamiltonian-MPO
contraction while avoiding a complete MPS sweep for every Hamiltonian term.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable, NamedTuple, Sequence

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


Array = jax.Array


class ThermodynamicValues(NamedTuple):
    """JAX scalar totals F and E, entropy in nats, norm, and minimum weight.

    ``minimum_schmidt_weight`` is a regularized normalized spectrum minimum;
    structural zero modes can make it zero even for a valid state.
    """

    free_energy: Array
    energy: Array
    entropy_nats: Array
    norm: Array
    minimum_schmidt_weight: Array


def _validated_integer(name: str, value: object, minimum: int) -> int:
    """Return an integer at least ``minimum``, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, found {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, found {result}")
    return result


def _validated_real(name: str, value: object) -> float:
    """Return a finite real scalar as float, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, found {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, found {result}")
    return result


@dataclass(frozen=True)
class JAXMPSObjectiveSpec:
    """Static, pickle-safe definition of an exact-MPS Gibbs objective.

    ``edges`` index system qubits from ``0`` through ``num_system - 1``.
    With the default ``ansatz_type='contiguous'``, ancillas precede system
    qubits in the MPS ordering.  ``ansatz_type='interleaved_pair'`` uses the
    distributed pair-explicit ordering.  Each edge is stored in ascending
    logical-system order; duplicate undirected edges and self-edges are
    rejected.

    A spec may be passed to a spawned worker process.  JIT-compiled objective
    functions should instead be constructed inside each worker and are not
    part of the pickle-stability guarantee.
    """

    name: str
    num_system: int
    num_ancillas: int
    num_layers: int
    edges: tuple[tuple[int, int], ...] = ()
    beta: float = 1.0
    j: float = -1.0
    hz: float = 0.0
    hx: float = -0.5
    ansatz_type: str = "contiguous"

    def __post_init__(self) -> None:
        """Validate dimensions and signed couplings, and canonicalize edges.

        Edges retain input order but each pair is sorted. Duplicate undirected
        edges, self-edges, invalid sites, and nonpositive beta are rejected.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())

        num_system = _validated_integer("num_system", self.num_system, 1)
        num_ancillas = _validated_integer(
            "num_ancillas", self.num_ancillas, 0
        )
        num_layers = _validated_integer("num_layers", self.num_layers, 1)
        object.__setattr__(self, "num_system", num_system)
        object.__setattr__(self, "num_ancillas", num_ancillas)
        object.__setattr__(self, "num_layers", num_layers)

        beta = _validated_real("beta", self.beta)
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, found {beta}")
        object.__setattr__(self, "beta", beta)
        for coefficient in ("j", "hz", "hx"):
            object.__setattr__(
                self,
                coefficient,
                _validated_real(coefficient, getattr(self, coefficient)),
            )

        if not isinstance(self.ansatz_type, str):
            raise TypeError("ansatz_type must be a string")
        ansatz_type = self.ansatz_type.strip().lower()
        if ansatz_type not in {"contiguous", "interleaved_pair"}:
            raise ValueError(
                "ansatz_type must be 'contiguous' or 'interleaved_pair', "
                f"found {self.ansatz_type!r}"
            )
        object.__setattr__(self, "ansatz_type", ansatz_type)
        if ansatz_type == "interleaved_pair" and num_ancillas > num_system:
            raise ValueError(
                "interleaved_pair requires num_ancillas <= num_system"
            )

        try:
            edge_values = tuple(self.edges)
        except TypeError as exc:
            raise TypeError("edges must be an iterable of two-site pairs") from exc

        canonical_edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for edge_index, edge in enumerate(edge_values):
            try:
                pair = tuple(edge)
            except TypeError as exc:
                raise TypeError(
                    f"edge {edge_index} must be a two-site pair"
                ) from exc
            if len(pair) != 2:
                raise ValueError(
                    f"edge {edge_index} must contain exactly two sites"
                )
            first = _validated_integer(
                f"edges[{edge_index}][0]", pair[0], 0
            )
            second = _validated_integer(
                f"edges[{edge_index}][1]", pair[1], 0
            )
            if first >= num_system or second >= num_system:
                raise ValueError(
                    f"edge {edge_index}={pair!r} is outside a "
                    f"{num_system}-site system"
                )
            if first == second:
                raise ValueError(f"edge {edge_index} is a self-edge at {first}")
            canonical = (min(first, second), max(first, second))
            if canonical in seen:
                raise ValueError(f"duplicate undirected edge {canonical}")
            seen.add(canonical)
            canonical_edges.append(canonical)
        object.__setattr__(self, "edges", tuple(canonical_edges))

    @property
    def num_qubits(self) -> int:
        """Return the total number of system and ancilla qubits."""
        return self.num_system + self.num_ancillas

    @property
    def num_parameters(self) -> int:
        """Return the number of RY angles: one per qubit per layer."""
        return self.num_qubits * self.num_layers

    @property
    def paired_system_indices(self) -> tuple[int, ...]:
        """Return paired logical system indices, or empty for contiguous HEA."""

        if self.num_ancillas == 0:
            return ()
        if self.ansatz_type == "contiguous":
            # The original ladder does not define explicit purification pairs;
            # this metadata is meaningful only for the interleaved variant.
            return ()
        return tuple(
            (ancilla * self.num_system) // self.num_ancillas
            for ancilla in range(self.num_ancillas)
        )

    @property
    def ancilla_positions(self) -> tuple[int, ...]:
        """Return zero-based MPS positions occupied by ancillas."""

        if self.ansatz_type == "contiguous":
            return tuple(range(self.num_ancillas))
        return tuple(
            system + ancilla
            for ancilla, system in enumerate(self.paired_system_indices)
        )

    @property
    def system_positions(self) -> tuple[int, ...]:
        """Return zero-based MPS positions in logical system-site order."""

        if self.ansatz_type == "contiguous":
            return tuple(
                self.num_ancillas + system
                for system in range(self.num_system)
            )
        pairs = self.paired_system_indices
        return tuple(
            system + sum(pair <= system for pair in pairs)
            for system in range(self.num_system)
        )

    @property
    def pair_positions(self) -> tuple[tuple[int, int], ...]:
        """Return interleaved ancilla/system MPS-position pairs, else empty."""

        if self.ansatz_type != "interleaved_pair":
            return ()
        return tuple(
            (ancilla_position, self.system_positions[system])
            for ancilla_position, system in zip(
                self.ancilla_positions, self.paired_system_indices
            )
        )


def chain_edges(length: int) -> tuple[tuple[int, int], ...]:
    """Return open-boundary nearest-neighbour edges for a chain."""

    length = _validated_integer("length", length, 1)
    return tuple((site, site + 1) for site in range(length - 1))


def square_edges(length: int) -> tuple[tuple[int, int], ...]:
    """Return row-major open-boundary edges for a square lattice."""

    length = _validated_integer("length", length, 1)
    edges: list[tuple[int, int]] = []
    for y in range(length):
        for x in range(length):
            site = x + y * length
            if x + 1 < length:
                edges.append((site, site + 1))
            if y + 1 < length:
                edges.append((site, site + length))
    return tuple(edges)


_IDENTITY = jnp.eye(2, dtype=jnp.float64)
_PAULI_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)
_PAULI_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.float64)
_CNOT_LEFT = jnp.asarray(
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ],
    dtype=jnp.float64,
)
_CNOT_RIGHT = jnp.stack((_IDENTITY, _PAULI_X), axis=0)


def _ry(theta: Array) -> Array:
    """Return the real 2-by-2 Y-rotation matrix for an angle in radians."""
    cosine = jnp.cos(theta / 2.0)
    sine = jnp.sin(theta / 2.0)
    return jnp.stack(
        (
            jnp.stack((cosine, -sine)),
            jnp.stack((sine, cosine)),
        )
    )


def _apply_one_site(tensor: Array, gate: Array) -> Array:
    """Apply a 2-by-2 gate to the physical axis of a (left, site, right) tensor."""
    return jnp.einsum("pq,lqr->lpr", gate, tensor, optimize=True)


def _apply_cnot(left: Array, right: Array) -> tuple[Array, Array]:
    """Apply an exact left-control CNOT and double the shared tensor bond."""

    left_tmp = jnp.einsum("kpq,lqr->lkpr", _CNOT_LEFT, left, optimize=True)
    left_tmp = jnp.transpose(left_tmp, (0, 2, 3, 1))
    left_new = jnp.reshape(
        left_tmp,
        (left.shape[0], 2, left.shape[2] * 2),
    )

    right_tmp = jnp.einsum(
        "kpq,lqr->lkpr", _CNOT_RIGHT, right, optimize=True
    )
    right_new = jnp.reshape(
        right_tmp,
        (right.shape[0] * 2, 2, right.shape[2]),
    )
    return left_new, right_new


def _apply_cnot_range_two(
    left: Array, middle: Array, right: Array
) -> tuple[Array, Array, Array]:
    """Apply an exact CNOT across one intervening MPS site.

    The CNOT's rank-two operator-Schmidt label is attached to the control,
    propagated diagonally through the middle tensor, and consumed at the
    target.  This is an exact three-site/general-MPS update in already
    refactorized form: it doubles the two crossed bonds, performs no SWAPs,
    and avoids the singular-vector gradient ambiguity of a numerical SVD.
    """

    left_tmp = jnp.einsum(
        "kpq,lqr->lkpr", _CNOT_LEFT, left, optimize=True
    )
    left_tmp = jnp.transpose(left_tmp, (0, 2, 3, 1))
    left_new = jnp.reshape(
        left_tmp, (left.shape[0], 2, left.shape[2] * 2)
    )

    bridge = jnp.eye(2, dtype=middle.dtype)
    middle_tmp = jnp.einsum(
        "lpr,km->lkprm", middle, bridge, optimize=True
    )
    middle_new = jnp.reshape(
        middle_tmp,
        (middle.shape[0] * 2, 2, middle.shape[2] * 2),
    )

    right_tmp = jnp.einsum(
        "kpq,lqr->lkpr", _CNOT_RIGHT, right, optimize=True
    )
    right_new = jnp.reshape(
        right_tmp, (right.shape[0] * 2, 2, right.shape[2])
    )
    return left_new, middle_new, right_new


def _validated_parameters(params: Array, spec: JAXMPSObjectiveSpec) -> Array:
    """Convert parameters to float64 and check the flat vector's length.

    Values are not checked for finiteness here, so this shape-only validation
    can execute while JAX traces the parameter array.
    """
    array = jnp.asarray(params, dtype=jnp.float64)
    if array.ndim != 1:
        raise ValueError(
            f"params must be rank one, found shape {array.shape!r}"
        )
    if array.shape[0] != spec.num_parameters:
        raise ValueError(
            f"expected {spec.num_parameters} parameters, found "
            f"{array.shape[0]}"
        )
    return array


def _build_exact_mps_states(
    params: Array, spec: JAXMPSObjectiveSpec
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """Build the final MPS and the MPS used for its ancilla entropy.

    For the interleaved ansatz, the final system-only CNOT staircase is a
    local unitary with respect to the ancilla/system bipartition.  The ancilla
    density matrix is therefore exactly unchanged by that staircase.  Saving
    the tensors immediately before it gives the entropy contraction smaller
    bonds, while the returned final tensors still include every gate and are
    used for the energy.  The contiguous ansatz uses the same final tensors for
    both quantities.
    """

    parameter_layers = jnp.reshape(
        _validated_parameters(params, spec),
        (spec.num_layers, spec.num_qubits),
    )
    zero = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    tensors: list[Array] = [
        jnp.reshape(zero, (1, 2, 1)) for _ in range(spec.num_qubits)
    ]
    entropy_tensors: tuple[Array, ...] | None = None

    for layer in range(spec.num_layers):
        for site in range(spec.num_qubits):
            tensors[site] = _apply_one_site(
                tensors[site], _ry(parameter_layers[layer, site])
            )
        if spec.ansatz_type == "contiguous":
            for site in range(spec.num_qubits - 1):
                tensors[site], tensors[site + 1] = _apply_cnot(
                    tensors[site], tensors[site + 1]
                )
        else:
            for ancilla_position, system_position in spec.pair_positions:
                if system_position != ancilla_position + 1:
                    raise AssertionError("interleaved pair sites are not adjacent")
                tensors[ancilla_position], tensors[system_position] = (
                    _apply_cnot(
                        tensors[ancilla_position], tensors[system_position]
                    )
                )
            if layer == spec.num_layers - 1:
                entropy_tensors = tuple(tensors)
            for system in range(spec.num_system - 1):
                first = spec.system_positions[system]
                second = spec.system_positions[system + 1]
                if second == first + 1:
                    tensors[first], tensors[second] = _apply_cnot(
                        tensors[first], tensors[second]
                    )
                elif second == first + 2:
                    (
                        tensors[first],
                        tensors[first + 1],
                        tensors[second],
                    ) = _apply_cnot_range_two(
                        tensors[first], tensors[first + 1], tensors[second]
                    )
                else:
                    raise AssertionError(
                        "logical neighbours must be adjacent or range two"
                    )
    final_tensors = tuple(tensors)
    if entropy_tensors is None:
        entropy_tensors = final_tensors
    return final_tensors, entropy_tensors


def _balanced_ancilla_cut(
    tensors: Sequence[Array], spec: JAXMPSObjectiveSpec
) -> int:
    """Choose a cut by ancilla balance, intermediate size, then centrality.

    Return the number of sites on the left; equal scores use the smaller cut.
    """

    ancillas = set(spec.ancilla_positions)
    candidates: list[tuple[int, int, float, int]] = []
    for cut in range(spec.num_qubits + 1):
        left_count = sum(position < cut for position in ancillas)
        right_count = spec.num_ancillas - left_count
        if cut == 0:
            bond = int(tensors[0].shape[0])
        elif cut == spec.num_qubits:
            bond = int(tensors[-1].shape[2])
        else:
            left_bond = int(tensors[cut - 1].shape[2])
            right_bond = int(tensors[cut].shape[0])
            if left_bond != right_bond:
                raise ValueError("inconsistent MPS bond dimensions")
            bond = left_bond
        largest_intermediate = max(
            (4**left_count) * bond**2,
            (4**right_count) * bond**2,
        )
        candidates.append(
            (
                abs(left_count - right_count),
                largest_intermediate,
                abs(cut - spec.num_qubits / 2.0),
                cut,
            )
        )
    return min(candidates)[-1]


def _rho_from_open_pairs(open_pairs: Array, num_ancillas: int) -> Array:
    """Group site-ordered ket/bra pairs into a ``2**N_a``-square matrix."""

    if num_ancillas == 0:
        return jnp.reshape(open_pairs, (1, 1))
    pair_axes = jnp.reshape(open_pairs, (2, 2) * num_ancillas)
    permutation = tuple(range(0, 2 * num_ancillas, 2)) + tuple(
        range(1, 2 * num_ancillas, 2)
    )
    dimension = 2**num_ancillas
    return jnp.reshape(jnp.transpose(pair_axes, permutation), (dimension, dimension))


def ancilla_density_matrix_balanced(
    tensors: Sequence[Array], spec: JAXMPSObjectiveSpec
) -> Array:
    """Construct interleaved ``rho_A`` with a balanced double-layer sweep.

    The left and right intermediates expose at most about half of the ancilla
    ket/bra pairs before the shared doubled MPS bond is contracted.  All
    operations remain in JAX so the result can be differentiated and JIT
    compiled.

    Args:
        tensors: MPS tensors with axes ``(left bond, physical site, right bond)``.
        spec: Interleaved-layout specification defining the ancilla positions.

    Returns:
        Hermitized, trace-normalized ancilla density matrix of shape
        ``(2**num_ancillas, 2**num_ancillas)`` in ancilla site order.
    """

    if spec.ansatz_type != "interleaved_pair":
        raise ValueError("balanced rho_A is defined for interleaved_pair only")
    cut = _balanced_ancilla_cut(tensors, spec)
    ancillas = set(spec.ancilla_positions)

    left = jnp.ones((1, 1, 1), dtype=tensors[0].dtype)
    for site in range(cut):
        tensor = tensors[site]
        if site in ancillas:
            left = jnp.einsum(
                "plm,lar,mbs->pabrs",
                left,
                tensor,
                jnp.conj(tensor),
                optimize=True,
            )
            left = jnp.reshape(
                left, (left.shape[0] * 4, tensor.shape[2], tensor.shape[2])
            )
        else:
            left = jnp.einsum(
                "plm,lqr,mqs->prs",
                left,
                tensor,
                jnp.conj(tensor),
                optimize=True,
            )

    right = jnp.ones((1, 1, 1), dtype=tensors[-1].dtype)
    for site in range(spec.num_qubits - 1, cut - 1, -1):
        tensor = tensors[site]
        if site in ancillas:
            right = jnp.einsum(
                "lar,mbs,rsp->lmabp",
                tensor,
                jnp.conj(tensor),
                right,
                optimize=True,
            )
            right = jnp.reshape(
                right, (tensor.shape[0], tensor.shape[0], 4 * right.shape[-1])
            )
        else:
            right = jnp.einsum(
                "lqr,mqs,rsp->lmp",
                tensor,
                jnp.conj(tensor),
                right,
                optimize=True,
            )

    open_pairs = jnp.einsum("plm,lmq->pq", left, right, optimize=True)
    rho = _rho_from_open_pairs(open_pairs, spec.num_ancillas)
    rho = (rho + jnp.conj(rho.T)) / 2.0
    return rho / jnp.real(jnp.trace(rho))


_ENTROPY_WEIGHT_THRESHOLD = 1.0e-14


def _density_matrix_entropy_eigensystem(
    rho: Array,
) -> tuple[tuple[Array, Array], tuple[Array, Array, Array, Array, Array]]:
    """Evaluate the regularized entropy and retain its spectral VJP data."""

    raw_weights, vectors = jnp.linalg.eigh(
        rho,
        UPLO="L",
        symmetrize_input=False,
    )
    threshold = jnp.asarray(_ENTROPY_WEIGHT_THRESHOLD, dtype=raw_weights.dtype)
    raw_mask = raw_weights > threshold
    truncated_weights = jnp.where(raw_mask, raw_weights, 0.0)
    normalizer = jnp.sum(truncated_weights)
    weights = truncated_weights / normalizer
    weight_mask = weights > threshold
    safe_weights = jnp.where(weight_mask, weights, 1.0)
    entropy = -jnp.sum(
        jnp.where(weight_mask, weights * jnp.log(safe_weights), 0.0)
    )
    outputs = entropy, jnp.min(weights)
    residual = vectors, weights, raw_mask, weight_mask, normalizer
    return outputs, residual


@jax.custom_vjp
def _entropy_nats_from_density_matrix(rho: Array) -> tuple[Array, Array]:
    """Return regularized natural-log entropy and minimum normalized weight.

    The density-matrix callers Hermitize and normalize their results before
    calling this helper.  Bypassing ``eigh``'s default input
    symmetrization therefore avoids materializing a second dense
    ``2**N_a``-square matrix while retaining the same Hermitian eigensystem.
    ``eigh`` is used rather than ``eigvalsh`` because the supported JAX API
    exposes ``symmetrize_input`` only on ``eigh``.

    Eigenvalues at or below ``_ENTROPY_WEIGHT_THRESHOLD`` are discarded and
    the retained spectrum is renormalized.  This is an explicit numerical
    regularization: its derivative is zero in discarded modes and is not the
    singular derivative of the unregularized von Neumann entropy at the
    boundary of the density-matrix cone.
    """

    outputs, _ = _density_matrix_entropy_eigensystem(rho)
    return outputs


def _entropy_nats_from_density_matrix_fwd(
    rho: Array,
) -> tuple[tuple[Array, Array], tuple[Array, Array, Array, Array, Array]]:
    """Return entropy outputs and retained eigensystem data for the custom VJP."""
    return _density_matrix_entropy_eigensystem(rho)


def _entropy_nats_from_density_matrix_bwd(
    residual: tuple[Array, Array, Array, Array, Array],
    cotangents: tuple[Array, Array],
) -> tuple[Array]:
    """Apply the masked, renormalized spectral VJP without eigenvector gaps.

    The returned cotangent follows JAX's complex-array convention. Tied
    minimum normalized weights share their minimum-value cotangent equally.
    """

    vectors, weights, raw_mask, weight_mask, normalizer = residual
    entropy_cotangent, minimum_cotangent = cotangents
    safe_weights = jnp.where(weight_mask, weights, 1.0)

    entropy_weight_gradient = jnp.where(
        weight_mask,
        -(jnp.log(safe_weights) + 1.0),
        0.0,
    )
    entropy_mean = jnp.sum(weights * entropy_weight_gradient)
    entropy_spectral_gradient = jnp.where(
        raw_mask,
        (entropy_weight_gradient - entropy_mean) / normalizer,
        0.0,
    )

    minimum_weight = jnp.min(weights)
    minimum_mask = weights == minimum_weight
    minimum_weight_gradient = minimum_mask / jnp.sum(minimum_mask)
    minimum_mean = jnp.sum(weights * minimum_weight_gradient)
    minimum_spectral_gradient = jnp.where(
        raw_mask,
        (minimum_weight_gradient - minimum_mean) / normalizer,
        0.0,
    )

    spectral_cotangent = (
        entropy_cotangent * entropy_spectral_gradient
        + minimum_cotangent * minimum_spectral_gradient
    )
    rho_cotangent = (
        vectors * spectral_cotangent[jnp.newaxis, :]
    ) @ jnp.conj(vectors.T)
    rho_cotangent = (rho_cotangent + jnp.conj(rho_cotangent.T)) / 2.0
    # JAX represents the reverse-mode cotangent of a real scalar with
    # respect to a complex array using the conjugate (equivalently, for this
    # Hermitian matrix, the transpose) of the ordinary matrix differential.
    # The previous real-valued MPS callers made this distinction invisible.
    return (jnp.conj(rho_cotangent),)


_entropy_nats_from_density_matrix.defvjp(
    _entropy_nats_from_density_matrix_fwd,
    _entropy_nats_from_density_matrix_bwd,
)


def _left_transfer(environment: Array, tensor: Array, operator: Array) -> Array:
    """Advance a doubled left-bond environment through one real MPS tensor.

    Insert the 2-by-2 physical operator and return the doubled right-bond
    environment. Both tensor copies are real; no conjugation is performed.
    """
    return jnp.einsum(
        "ab,apr,bqs,qp->rs",
        environment,
        tensor,
        tensor,
        operator,
        optimize=True,
    )


def _right_transfer(environment: Array, tensor: Array, operator: Array) -> Array:
    """Advance a doubled right-bond environment leftward through a real tensor.

    Insert the 2-by-2 physical operator and return the doubled left-bond
    environment, using the same real-tensor convention as ``_left_transfer``.
    """
    return jnp.einsum(
        "apr,bqs,rs,qp->ab",
        tensor,
        tensor,
        environment,
        operator,
        optimize=True,
    )


def _identity_environments(
    tensors: Sequence[Array],
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    """Return identity-transfer environments at every cut of a real MPS.

    Both sequences have ``len(tensors) + 1`` entries. For site k, ``left[k]``
    precedes its tensor and ``right[k + 1]`` follows it.
    """

    left: list[Array] = [jnp.ones((1, 1), dtype=jnp.float64)]
    for tensor in tensors:
        left.append(_left_transfer(left[-1], tensor, _IDENTITY))

    right_reversed: list[Array] = [
        jnp.ones((1, 1), dtype=jnp.float64)
    ]
    for tensor in reversed(tensors):
        right_reversed.append(
            _right_transfer(right_reversed[-1], tensor, _IDENTITY)
        )
    return tuple(left), tuple(reversed(right_reversed))


def _environment_overlap(left: Array, right: Array) -> Array:
    """Contract matching doubled-bond environments to an unnormalized scalar."""
    return jnp.einsum("ab,ab->", left, right, optimize=True)


def _one_site_numerator(
    tensors: Sequence[Array],
    left: Sequence[Array],
    right: Sequence[Array],
    site: int,
    operator: Array,
) -> Array:
    """Contract an unnormalized one-site expectation with real-MPS environments."""
    inserted = _left_transfer(left[site], tensors[site], operator)
    return _environment_overlap(inserted, right[site + 1])


def _two_site_numerator(
    tensors: Sequence[Array],
    left: Sequence[Array],
    right: Sequence[Array],
    first: int,
    second: int,
    first_operator: Array,
    second_operator: Array,
) -> Array:
    """Contract an unnormalized two-site expectation with ``first < second``.

    An identity-transfer bridge spans any MPS sites between the operators.
    """
    inserted = _left_transfer(left[first], tensors[first], first_operator)
    for site in range(first + 1, second):
        inserted = _left_transfer(inserted, tensors[site], _IDENTITY)
    inserted = _left_transfer(
        inserted, tensors[second], second_operator
    )
    return _environment_overlap(inserted, right[second + 1])


def _entropy_nats_from_grams(
    left_gram: Array, right_gram: Array
) -> tuple[Array, Array]:
    """Return regularized cut entropy and minimum weight from real Gram matrices.

    A ``1e-14`` diagonal stabilizer protects the left-Gram Cholesky. Negative
    spectral weights are clipped, then normalized; entropy contributions at
    weights at or below ``1e-15`` are omitted. This is distinct from the
    density-matrix helper's truncation and custom-VJP regularization.
    """

    left_gram = (left_gram + left_gram.T) / 2.0
    right_gram = (right_gram + right_gram.T) / 2.0
    dimension = left_gram.shape[0]
    # The stabilizer protects Cholesky from float64 roundoff.  It is far below
    # the numerical tolerance used for endpoint validation.
    stabilizer = 1.0e-14 * jnp.eye(dimension, dtype=jnp.float64)
    cholesky = jnp.linalg.cholesky(left_gram + stabilizer)
    spectrum_matrix = cholesky.T @ right_gram @ cholesky
    spectrum_matrix = (spectrum_matrix + spectrum_matrix.T) / 2.0
    weights = jnp.linalg.eigvalsh(spectrum_matrix)
    weights = jnp.clip(weights, 0.0)
    weights = weights / jnp.sum(weights)
    safe_weights = jnp.maximum(weights, 1.0e-15)
    entropy = -jnp.sum(
        jnp.where(
            weights > 1.0e-15,
            weights * jnp.log(safe_weights),
            0.0,
        )
    )
    return entropy, jnp.min(weights)


def thermodynamics(
    params: Array, spec: JAXMPSObjectiveSpec
) -> ThermodynamicValues:
    """Evaluate ``(F, E, S_nats, norm, minimum Schmidt weight)``.

    The function is pure JAX and may be differentiated or enclosed in a
    larger ``jax.jit``.  ``spec`` is static Python configuration, while the
    one-dimensional parameter vector is the sole dynamic argument.

    Args:
        params: Flat layer-major RY angles, with sites in circuit/MPS order.
        spec: Static lattice, layout, couplings, and positive physical beta.

    Returns:
        Scalar total ``F = E - S/beta``, total energy, entropy in nats,
        squared state norm, and the regularized minimum spectral weight.
        The optimizer, not this function, rescales F to ``beta*F/N``.
    """

    if not isinstance(spec, JAXMPSObjectiveSpec):
        raise TypeError("spec must be a JAXMPSObjectiveSpec")
    tensors, entropy_tensors = _build_exact_mps_states(params, spec)
    left, right = _identity_environments(tensors)
    norm = jnp.reshape(left[-1], ())
    system_positions = spec.system_positions

    energy = jnp.asarray(0.0, dtype=jnp.float64)
    if spec.hx != 0.0:
        for system_site in range(spec.num_system):
            site = system_positions[system_site]
            numerator = _one_site_numerator(
                tensors, left, right, site, _PAULI_X
            )
            energy = energy + spec.hx * (numerator / norm)
    if spec.hz != 0.0:
        for system_site in range(spec.num_system):
            site = system_positions[system_site]
            numerator = _one_site_numerator(
                tensors, left, right, site, _PAULI_Z
            )
            energy = energy + spec.hz * (numerator / norm)
    if spec.j != 0.0:
        for first, second in spec.edges:
            numerator = _two_site_numerator(
                tensors,
                left,
                right,
                system_positions[first],
                system_positions[second],
                _PAULI_Z,
                _PAULI_Z,
            )
            energy = energy + spec.j * (numerator / norm)

    if spec.ansatz_type == "contiguous":
        entropy, minimum_weight = _entropy_nats_from_grams(
            left[spec.num_ancillas], right[spec.num_ancillas]
        )
    else:
        if spec.num_layers >= 3:
            # The entropy eigensystem must remain live for the reverse pass,
            # but the balanced contraction's intermediate open-pair
            # environments do not.  At L >= 3 those environments are large
            # enough that rematerializing this inexpensive contraction during
            # backward measurably lowers peak memory.  At L=2 rematerialization
            # lowered the gradient-stage footprint but not the process-wide
            # peak and cost runtime, so its direct path keeps the faster
            # tradeoff.
            rho_ancilla = jax.checkpoint(
                lambda inner_tensors: ancilla_density_matrix_balanced(
                    inner_tensors, spec
                )
            )(entropy_tensors)
        else:
            rho_ancilla = ancilla_density_matrix_balanced(
                entropy_tensors, spec
            )
        entropy, minimum_weight = _entropy_nats_from_density_matrix(rho_ancilla)
    free_energy = energy - entropy / spec.beta
    return ThermodynamicValues(
        free_energy,
        energy,
        entropy,
        norm,
        minimum_weight,
    )


def make_loss_with_aux(
    spec: JAXMPSObjectiveSpec,
) -> Callable[[Array], tuple[Array, tuple[Array, Array, Array, Array]]]:
    """Return an uncompiled total-F loss with ``(E, S, norm, min_weight)`` aux.

    Entropy is in nats; ``spec`` is captured as static Python configuration.
    """

    if not isinstance(spec, JAXMPSObjectiveSpec):
        raise TypeError("spec must be a JAXMPSObjectiveSpec")

    def loss_with_aux(
        params: Array,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        """Evaluate total F and the four physical diagnostics for one vector."""
        values = thermodynamics(params, spec)
        return values.free_energy, (
            values.energy,
            values.entropy_nats,
            values.norm,
            values.minimum_schmidt_weight,
        )

    return loss_with_aux


__all__ = [
    "JAXMPSObjectiveSpec",
    "ThermodynamicValues",
    "ancilla_density_matrix_balanced",
    "chain_edges",
    "make_loss_with_aux",
    "square_edges",
    "thermodynamics",
]
