"""Parity-Lanczos and zero-temperature covariance ground-state kernels.

Release orchestration lives in run_references.
The Hamiltonian is -J sum_<ij> Z_i Z_j - h sum_i X_i with Pauli operators
and open boundaries. Pure-state beta envelopes are ground-state comparators,
not thermal Gibbs-state predictions.
"""
from __future__ import annotations
import hashlib
import math
from types import ModuleType
from typing import Any, Iterable, Sequence
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

COUPLING = 1.0
FIELD = 0.5
LANCZOS_TOLERANCE = 1.0e-13
LANCZOS_MAXITER = 10_000
LANCZOS_NCV = 40
LOW_LEVELS_PER_PARITY = 3


def system_edges(geometry: str, nx: int, ny: int = 1) -> list[tuple[int, int]]:
    """Return open nearest-neighbor edges for a chain or row-major square.

    Raises:
        ValueError: The geometry is unsupported or its dimensions are invalid.
    """
    if geometry == "chain":
        if ny != 1 or nx < 2:
            raise ValueError((geometry, nx, ny))
        return [(site, site + 1) for site in range(nx - 1)]
    if geometry != "square" or nx != ny or nx < 2:
        raise ValueError((geometry, nx, ny))
    edges: list[tuple[int, int]] = []
    for row in range(ny):
        for column in range(nx):
            site = column + row * nx
            if column + 1 < nx:
                edges.append((site, site + 1))
            if row + 1 < ny:
                edges.append((site, site + nx))
    return edges


def zz_diagonal(
    basis: np.ndarray, edges: Sequence[tuple[int, int]], coupling: float
) -> np.ndarray:
    """Evaluate -J sum_<ij> Z_i Z_j on integer computational-basis labels."""
    diagonal = np.zeros(basis.size, dtype=np.float64)
    for first, second in edges:
        unequal = (((basis >> first) ^ (basis >> second)) & 1).astype(np.float64)
        diagonal -= coupling * (1.0 - 2.0 * unequal)
    return diagonal


def full_hamiltonian_csr(
    num_spins: int,
    edges: Sequence[tuple[int, int]],
    coupling: float = COUPLING,
    field: float = FIELD,
) -> csr_matrix:
    """Build the full Pauli TFIM Hamiltonian with site zero as the low bit."""
    dimension = 1 << num_spins
    basis = np.arange(dimension, dtype=np.int64)
    blocks = num_spins + 1
    rows = np.tile(basis, blocks)
    columns = np.empty(blocks * dimension, dtype=np.int64)
    values = np.empty(blocks * dimension, dtype=np.float64)
    columns[:dimension] = basis
    values[:dimension] = zz_diagonal(basis, edges, coupling)
    for site in range(num_spins):
        start = (site + 1) * dimension
        stop = start + dimension
        columns[start:stop] = basis ^ (1 << site)
        values[start:stop] = -field
    matrix = csr_matrix(
        (values, (rows, columns)), shape=(dimension, dimension), dtype=np.float64
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def parity_sector_operator(
    num_spins: int,
    edges: Sequence[tuple[int, int]],
    parity: int,
    coupling: float = COUPLING,
    field: float = FIELD,
) -> LinearOperator:
    """Return H in the P=prod_i X_i parity sector.

    Representatives have their highest bit fixed to zero and
    |r,p>=(|r>+p|complement(r)>)/sqrt(2).  Flipping the highest-bit spin
    complements all lower representative bits and contributes the sector sign.
    """

    if parity not in (-1, 1):
        raise ValueError(parity)
    dimension = 1 << (num_spins - 1)
    representatives = np.arange(dimension, dtype=np.int64)
    diagonal = zz_diagonal(representatives, edges, coupling)
    lower_flips = tuple(
        representatives ^ (1 << site) for site in range(num_spins - 1)
    )
    highest_flip = representatives ^ (dimension - 1)

    def matvec(vector: np.ndarray) -> np.ndarray:
        """Apply the fixed-parity Hamiltonian to real representative amplitudes."""
        vector = np.asarray(vector, dtype=np.float64)
        result = diagonal * vector
        for flipped in lower_flips:
            result = result - field * vector[flipped]
        result = result - field * parity * vector[highest_flip]
        return result

    return LinearOperator(
        shape=(dimension, dimension), matvec=matvec, rmatvec=matvec, dtype=np.float64
    )


def deterministic_v0(dimension: int, label: str) -> np.ndarray:
    """Return a normalized real Lanczos start vector seeded by an ASCII label."""
    digest = hashlib.sha256(label.encode("ascii")).digest()
    seed = int.from_bytes(digest[:8], "little")
    vector = np.random.Generator(np.random.PCG64(seed)).normal(size=dimension)
    vector /= np.linalg.norm(vector)
    return vector.astype(np.float64)


def solve_sector(
    num_spins: int,
    edges: Sequence[tuple[int, int]],
    parity: int,
    levels: int = LOW_LEVELS_PER_PARITY,
    seed_label: str = "primary",
) -> tuple[np.ndarray, np.ndarray]:
    """Find and sort the lowest levels in one spin-inversion parity sector.

    Returns:
        Ritz energies and corresponding real eigenvector columns in the
        representative basis, using the module's fixed J and h.
    """
    operator = parity_sector_operator(num_spins, edges, parity)
    values, vectors = eigsh(
        operator,
        k=levels,
        which="SA",
        tol=LANCZOS_TOLERANCE,
        maxiter=LANCZOS_MAXITER,
        ncv=min(LANCZOS_NCV, operator.shape[0] - 1),
        v0=deterministic_v0(operator.shape[0], f"{seed_label}|N={num_spins}|p={parity}"),
    )
    order = np.argsort(values)
    return np.asarray(values[order], dtype=np.float64), np.asarray(
        vectors[:, order], dtype=np.float64
    )


def expand_parity_vector(sector_vector: np.ndarray, parity: int) -> np.ndarray:
    """Expand representative amplitudes into a normalized full spin state.

    The caller supplies parity +1 or -1 for global spin inversion.
    """
    half_dimension = sector_vector.size
    num_spins = int(round(math.log2(2 * half_dimension)))
    if (1 << (num_spins - 1)) != half_dimension:
        raise ValueError("Sector dimension is not a power of two")
    representatives = np.arange(half_dimension, dtype=np.int64)
    complement = representatives ^ ((1 << num_spins) - 1)
    state = np.zeros(2 * half_dimension, dtype=np.float64)
    state[representatives] = sector_vector / math.sqrt(2.0)
    state[complement] = parity * sector_vector / math.sqrt(2.0)
    state /= np.linalg.norm(state)
    return state


def spin_basis_observables(
    state: np.ndarray,
    hamiltonian: csr_matrix,
    edges: Sequence[tuple[int, int]],
    coupling: float = COUPLING,
    field: float = FIELD,
) -> dict[str, Any]:
    """Evaluate Pauli observables and residuals for a real ground-state vector.

    The caller supplies a normalized eigenstate. Physical energy variance and
    entropy are recorded as zero for this pure eigenstate comparator; the
    squared eigenpair residual is retained separately as a numerical audit.
    """
    dimension = state.size
    num_spins = int(round(math.log2(dimension)))
    if (1 << num_spins) != dimension:
        raise ValueError("State dimension is not a power of two")
    basis = np.arange(dimension, dtype=np.int64)
    probabilities = np.square(state)
    bits = ((basis[:, None] >> np.arange(num_spins)) & 1).astype(np.float64)
    z_values = 1.0 - 2.0 * bits
    # ``einsum`` avoids spurious Accelerate/BLAS overflow warnings observed
    # for otherwise well-conditioned float64 dot products on macOS.
    correlations = np.einsum(
        "bi,bj,b->ij", z_values, z_values, probabilities, optimize=True
    )
    correlations = 0.5 * (correlations + correlations.T)
    magnetization_basis = np.sum(z_values, axis=1)
    magnetization = float(
        np.einsum("b,b->", probabilities, magnetization_basis, optimize=True)
    )
    magnetization_second = float(
        np.einsum(
            "b,b->", probabilities, np.square(magnetization_basis), optimize=True
        )
    )
    x_expectations = np.asarray(
        [
            float(
                np.einsum(
                    "b,b->", state, state[basis ^ (1 << site)], optimize=True
                )
            )
            for site in range(num_spins)
        ]
    )
    edge_correlations = np.asarray(
        [correlations[first, second] for first, second in edges], dtype=np.float64
    )
    interaction_energy = -coupling * float(np.sum(edge_correlations))
    transverse_energy = -field * float(np.sum(x_expectations))
    h_state = np.asarray(hamiltonian @ state, dtype=np.float64)
    energy = float(np.einsum("b,b->", state, h_state, optimize=True))
    residual = h_state - energy * state
    complement = basis ^ (dimension - 1)
    parity_expectation = float(
        np.einsum("b,b->", state, state[complement], optimize=True)
    )
    return {
        "energy": energy,
        "energy_density": energy / num_spins,
        # The exact eigenstate has zero physical energy variance.  Preserve
        # the finite solver residual separately as a numerical audit value.
        "energy_variance": 0.0,
        "numerical_energy_variance_residual": float(
            np.einsum("b,b->", residual, residual, optimize=True)
        ),
        "entropy_nats": 0.0,
        "entropy_density_nats": 0.0,
        "state_norm": float(np.einsum("b,b->", state, state, optimize=True)),
        "eigenpair_residual_l2": float(np.linalg.norm(residual)),
        "parity_expectation": parity_expectation,
        "mean_longitudinal_magnetization": magnetization,
        "magnetization_second_moment": magnetization_second,
        "magnetization_variance": magnetization_second - magnetization**2,
        "magnetization_variance_per_spin": (
            magnetization_second - magnetization**2
        )
        / num_spins,
        "transverse_magnetization_sum": float(np.sum(x_expectations)),
        "transverse_magnetization_by_site": x_expectations.tolist(),
        "nearest_neighbor_zz_sum": float(np.sum(edge_correlations)),
        "nearest_neighbor_zz_by_edge": edge_correlations.tolist(),
        "longitudinal_correlations": correlations.tolist(),
        "interaction_energy": interaction_energy,
        "transverse_field_energy": transverse_energy,
        "energy_decomposition_residual": interaction_energy
        + transverse_energy
        - energy,
        "correlation_symmetry_residual_max_abs": float(
            np.max(np.abs(correlations - correlations.T))
        ),
        "correlation_diagonal_residual_max_abs": float(
            np.max(np.abs(np.diag(correlations) - 1.0))
        ),
    }


def pure_state_envelope(
    energy: float, num_spins: int, betas: Iterable[float], variance_m_per_spin: float
) -> list[dict[str, float]]:
    """Evaluate a fixed pure-state comparator over a physical beta grid.

    Keep energy and zero entropy fixed while scaling the dimensionless
    objective and equal-time magnetic fluctuation by beta. These rows do not
    describe a finite-temperature Gibbs state or a Kubo susceptibility.
    """
    rows: list[dict[str, float]] = []
    for beta in betas:
        rows.append(
            {
                "beta": float(beta),
                "energy": energy,
                "energy_density": energy / num_spins,
                "entropy_nats": 0.0,
                "entropy_density_nats": 0.0,
                "free_energy": energy,
                "free_energy_density": energy / num_spins,
                "dimensionless_objective_C_beta": beta * energy / num_spins,
                "magnetization_fluctuation_chi_ET": beta * variance_m_per_spin,
                "energy_fluctuation_cV_fluc": 0.0,
            }
        )
    return rows


def zero_temperature_chain(
    module: ModuleType, num_spins: int, betas: Sequence[float]
) -> dict[str, Any]:
    """Construct exact open-chain ground observables from pure covariance.

    Args:
        module: Provider of Majorana-matrix and longitudinal-correlation kernels.
        num_spins: Chain length of at least two for the reported gap diagnostics.
        betas: Physical inverse temperatures for the pure-state comparator.

    Returns:
        A ground-state record with correlations, gaps, covariance checks, and
        a fixed-ground-state beta envelope, not a thermal reference.

    Raises:
        RuntimeError: A zero mode is unresolved or the mode count is incorrect.
    """
    majorana = np.asarray(module.majorana_matrix(num_spins, COUPLING, FIELD))
    eigenvalues, eigenvectors = np.linalg.eigh(1j * majorana)
    if float(np.min(np.abs(eigenvalues))) <= 1.0e-14:
        raise RuntimeError(f"Numerically unresolved zero mode for N={num_spins}")
    diagonal = 1j * np.sign(eigenvalues)
    transformed = np.einsum(
        "mk,k,nk->mn", eigenvectors, diagonal, eigenvectors.conj(), optimize=True
    )
    covariance = np.real_if_close(transformed, tol=1000).real
    covariance = 0.5 * (covariance - covariance.T)
    modes = np.sort(eigenvalues[eigenvalues > 0.0])
    if modes.size != num_spins:
        raise RuntimeError(f"Expected {num_spins} positive modes, got {modes.size}")
    correlations = np.eye(num_spins, dtype=np.float64)
    for first in range(num_spins):
        for second in range(first + 1, num_spins):
            value = module.longitudinal_correlation(covariance, first, second)
            correlations[first, second] = value
            correlations[second, first] = value
    x_expectations = -np.diag(covariance, k=1)[::2]
    edges = system_edges("chain", num_spins)
    edge_correlations = np.asarray(
        [correlations[first, second] for first, second in edges]
    )
    energy = -0.5 * float(np.sum(modes))
    interaction_energy = -COUPLING * float(np.sum(edge_correlations))
    transverse_energy = -FIELD * float(np.sum(x_expectations))
    magnetization_second = float(np.sum(correlations))
    variance_per_spin = magnetization_second / num_spins
    record = {
        "system_id": f"chain_N{num_spins}",
        "geometry": "open_chain",
        "nx": num_spins,
        "ny": 1,
        "num_spins": num_spins,
        "num_edges": len(edges),
        "hilbert_space_dimension": 1 << num_spins,
        "method": "exact zero-temperature free-fermion Majorana covariance",
        "energy": energy,
        "energy_density": energy / num_spins,
        "energy_variance": 0.0,
        "entropy_nats": 0.0,
        "entropy_density_nats": 0.0,
        "ground_state_parity": 1,
        "mean_longitudinal_magnetization": 0.0,
        "magnetization_second_moment": magnetization_second,
        "magnetization_variance": magnetization_second,
        "magnetization_variance_per_spin": variance_per_spin,
        "transverse_magnetization_sum": float(np.sum(x_expectations)),
        "transverse_magnetization_by_site": x_expectations.tolist(),
        "nearest_neighbor_zz_sum": float(np.sum(edge_correlations)),
        "nearest_neighbor_zz_by_edge": edge_correlations.tolist(),
        "longitudinal_correlations": correlations.tolist(),
        "interaction_energy": interaction_energy,
        "transverse_field_energy": transverse_energy,
        "energy_decomposition_residual": interaction_energy
        + transverse_energy
        - energy,
        "lowest_quasiparticle_energies": modes[:6].tolist(),
        "finite_size_parity_splitting": float(modes[0]),
        "gap_above_low_energy_parity_doublet": float(modes[1]),
        "lowest_same_parity_excitation_gap": float(modes[0] + modes[1]),
        "covariance_antisymmetry_residual_max_abs": float(
            np.max(np.abs(covariance + covariance.T))
        ),
        "covariance_purity_residual_max_abs": float(
            np.max(
                np.abs(
                    np.einsum(
                        "ik,kj->ij", covariance, covariance, optimize=True
                    )
                    + np.eye(2 * num_spins)
                )
            )
        ),
        "correlation_symmetry_residual_max_abs": float(
            np.max(np.abs(correlations - correlations.T))
        ),
        "correlation_diagonal_residual_max_abs": float(
            np.max(np.abs(np.diag(correlations) - 1.0))
        ),
    }
    record["pure_state_envelope"] = pure_state_envelope(
        energy, num_spins, betas, variance_per_spin
    )
    return record


def square_ground_state(
    side: int, betas: Sequence[float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute the even-parity ground comparator for an open TFIM square.

    Resolve low levels in both global-spin-inversion sectors and compare the
    even ground energy with an independent start and full-space Lanczos.

    Returns:
        The ground-state observable record and independent solver diagnostics.
        Diagnostics are returned for the orchestration layer to validate.
    """
    num_spins = side * side
    edges = system_edges("square", side, side)
    even_values, even_vectors = solve_sector(num_spins, edges, parity=1)
    odd_values, _ = solve_sector(num_spins, edges, parity=-1)
    state = expand_parity_vector(even_vectors[:, 0], parity=1)
    hamiltonian = full_hamiltonian_csr(num_spins, edges)
    observables = spin_basis_observables(state, hamiltonian, edges)
    merged_levels = sorted(
        [
            {"energy": float(value), "parity": parity, "sector_index": index}
            for parity, values in ((1, even_values), (-1, odd_values))
            for index, value in enumerate(values)
        ],
        key=lambda row: row["energy"],
    )
    independent_even, _ = solve_sector(
        num_spins, edges, parity=1, levels=1, seed_label="independent-repeat"
    )
    full_value, _ = eigsh(
        hamiltonian,
        k=1,
        which="SA",
        tol=LANCZOS_TOLERANCE,
        maxiter=LANCZOS_MAXITER,
        ncv=min(LANCZOS_NCV, hamiltonian.shape[0] - 1),
        v0=deterministic_v0(hamiltonian.shape[0], f"full-space|{side}x{side}"),
    )
    energy = observables["energy"]
    record: dict[str, Any] = {
        "system_id": f"square_{side}x{side}",
        "geometry": "open_square",
        "nx": side,
        "ny": side,
        "num_spins": num_spins,
        "num_edges": len(edges),
        "hilbert_space_dimension": 1 << num_spins,
        "method": "parity-resolved real sparse Lanczos",
        **observables,
        "ground_state_parity": 1,
        "low_lying_levels": merged_levels,
        "finite_size_parity_splitting": float(odd_values[0] - even_values[0]),
        "gap_above_low_energy_parity_doublet": float(
            min(even_values[1], odd_values[1]) - even_values[0]
        ),
        "lowest_same_parity_excitation_gap": float(
            even_values[1] - even_values[0]
        ),
        "pure_state_envelope": pure_state_envelope(
            energy,
            num_spins,
            betas,
            float(observables["magnetization_variance_per_spin"]),
        ),
    }
    checks = {
        "primary_even_ritz_energy": float(even_values[0]),
        "independent_even_ritz_energy": float(independent_even[0]),
        "independent_even_energy_difference_abs": float(
            abs(independent_even[0] - even_values[0])
        ),
        "full_hilbert_ritz_energy": float(full_value[0]),
        "full_hilbert_energy_difference_abs": float(
            abs(full_value[0] - even_values[0])
        ),
        "ritz_vs_rayleigh_difference_abs": float(abs(even_values[0] - energy)),
        "sparse_nnz": int(hamiltonian.nnz),
        "sparse_dtype": str(hamiltonian.dtype),
    }
    return record, checks
