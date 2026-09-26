#!/usr/bin/env python3
"""Exact finite-temperature references for the open 1D TFIM.

The Hamiltonian convention is the one used in the manuscript,

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i.

A global single-qubit rotation maps this to the standard free-fermion Ising
chain.  The routines below evaluate the energy, specific heat, equal-time
longitudinal fluctuation response, and correlations directly from the
thermal Majorana covariance matrix.  No variational data enter these
calculations. The reported susceptibility is beta Var(sum_i Z_i) / N,
not the Kubo susceptibility; Z and X are Pauli operators.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def majorana_matrix(num_spins: int, coupling: float = 1.0, field: float = 0.5) -> np.ndarray:
    """Return real antisymmetric A for the open-chain H=(i/4) a^T A a.

    Args:
        num_spins: Positive number of Pauli spins.
        coupling: Nearest-neighbor ZZ coupling J in the spin Hamiltonian.
        field: Transverse X field h in the spin Hamiltonian.
    """

    if num_spins < 1:
        raise ValueError("num_spins must be positive")
    matrix = np.zeros((2 * num_spins, 2 * num_spins), dtype=float)
    for site in range(num_spins):
        matrix[2 * site, 2 * site + 1] = 2.0 * field
    for site in range(num_spins - 1):
        matrix[2 * site + 1, 2 * site + 2] = 2.0 * coupling
    matrix -= matrix.T
    return matrix


def thermal_covariance(
    num_spins: int,
    beta: float,
    coupling: float = 1.0,
    field: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Majorana covariance Gamma and positive mode energies.

    Gamma[m,n] = i <a_m a_n> for m != n.  Diagonalizing the Hermitian matrix
    iA evaluates the thermal matrix function at physical inverse temperature
    beta. Numerically unresolved modes below the positive-energy cutoff are
    rejected rather than silently omitted.

    Raises:
        ValueError: beta is negative or num_spins is not positive.
        RuntimeError: The positive mode count differs from num_spins.
    """

    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    matrix = majorana_matrix(num_spins, coupling, field)
    eigenvalues, eigenvectors = np.linalg.eigh(1j * matrix)
    diagonal = 1j * np.tanh(0.5 * beta * eigenvalues)
    transformed = np.einsum(
        "mk,k,nk->mn", eigenvectors, diagonal, eigenvectors.conj(), optimize=True
    )
    covariance = np.real_if_close(transformed, tol=1000).real
    covariance = 0.5 * (covariance - covariance.T)
    positive_modes = np.sort(eigenvalues[eigenvalues > 1.0e-12])
    if positive_modes.size != num_spins:
        raise RuntimeError(
            f"Expected {num_spins} positive modes, found {positive_modes.size}"
        )
    return covariance, positive_modes


def pfaffian(matrix: np.ndarray, tolerance: float = 1.0e-14) -> complex:
    """Compute the Pfaffian of a skew-symmetric matrix by pivoted elimination."""

    work = np.asarray(matrix, dtype=complex).copy()
    if work.ndim != 2 or work.shape[0] != work.shape[1]:
        raise ValueError("Pfaffian input must be square")
    size = work.shape[0]
    if size % 2:
        return 0.0 + 0.0j
    if size == 0:
        return 1.0 + 0.0j
    if not np.allclose(work, -work.T, atol=1.0e-11, rtol=0.0):
        raise ValueError("Pfaffian input is not skew-symmetric")

    result = 1.0 + 0.0j
    for offset in range(0, size - 1, 2):
        pivot = offset + 1 + int(np.argmax(np.abs(work[offset, offset + 1 :])))
        if abs(work[offset, pivot]) < tolerance:
            return 0.0 + 0.0j
        if pivot != offset + 1:
            work[[offset + 1, pivot], :] = work[[pivot, offset + 1], :]
            work[:, [offset + 1, pivot]] = work[:, [pivot, offset + 1]]
            result *= -1.0
        value = work[offset, offset + 1]
        result *= value
        if offset + 2 < size:
            first = work[offset, offset + 2 :].copy()
            second = work[offset + 1, offset + 2 :].copy()
            block = work[offset + 2 :, offset + 2 :]
            block += (np.outer(second, first) - np.outer(first, second)) / value
            work[offset + 2 :, offset + 2 :] = block
    return result


def longitudinal_correlation(covariance: np.ndarray, site_i: int, site_j: int) -> float:
    """Return the Pauli correlation <Z_i Z_j> from Majorana covariance.

    Sites are zero-based; onsite correlations are returned as one directly.
    Offsite correlations use a Jordan-Wigner string evaluated by a Pfaffian.
    """

    if site_i == site_j:
        return 1.0
    if site_i > site_j:
        site_i, site_j = site_j, site_i
    num_spins = covariance.shape[0] // 2
    if site_i < 0 or site_j >= num_spins:
        raise IndexError((site_i, site_j))
    distance = site_j - site_i
    indices = np.arange(2 * site_i + 1, 2 * site_j + 1)
    two_point = -1j * covariance[np.ix_(indices, indices)]
    value = ((-1j) ** distance) * pfaffian(two_point)
    value = np.real_if_close(value, tol=1000)
    if np.iscomplexobj(value) and abs(value.imag) > 1.0e-9:
        raise FloatingPointError(f"Correlation has imaginary residual {value}")
    return float(np.real(value))


def exact_observables(
    num_spins: int,
    beta: float,
    coupling: float = 1.0,
    field: float = 0.5,
) -> dict[str, Any]:
    """Evaluate exact thermal energy and Pauli correlations for an open chain.

    Returns:
        A JSON-ready record containing energy moments, heat capacity per
        spin, all longitudinal correlations, and positive mode energies.
        ``susceptibility_per_spin`` is the equal-time fluctuation
        beta <(sum_i Z_i)^2> / N, not the Kubo susceptibility; the mean
        longitudinal magnetization vanishes by spin-inversion symmetry.
    """
    covariance, modes = thermal_covariance(num_spins, beta, coupling, field)
    occupations = np.tanh(0.5 * beta * modes)
    energy = -0.5 * float(np.sum(modes * occupations))
    energy_variance = 0.25 * float(
        np.sum(modes**2 / np.cosh(0.5 * beta * modes) ** 2)
    )
    correlations = np.eye(num_spins, dtype=float)
    for site_i in range(num_spins):
        for site_j in range(site_i + 1, num_spins):
            value = longitudinal_correlation(covariance, site_i, site_j)
            correlations[site_i, site_j] = value
            correlations[site_j, site_i] = value
    magnetization_second_moment = float(np.sum(correlations))
    susceptibility = beta * magnetization_second_moment / num_spins
    specific_heat = beta**2 * energy_variance / num_spins
    return {
        "num_spins": num_spins,
        "beta": beta,
        "energy": energy,
        "energy_density": energy / num_spins,
        "energy_variance": energy_variance,
        "specific_heat_per_spin": specific_heat,
        "magnetization_second_moment": magnetization_second_moment,
        "susceptibility_per_spin": susceptibility,
        "longitudinal_correlations": correlations.tolist(),
        "single_particle_energies": modes.tolist(),
    }
