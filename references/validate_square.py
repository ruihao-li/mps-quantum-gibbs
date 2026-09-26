#!/usr/bin/env python3
"""Independent, inexpensive gates for the symmetry-resolved thermal solver.

The reference Hamiltonian uses Kronecker products, not the sector builder.
No 4x4 eigendecomposition is run here. 2x2/3x3 complete spectra, projectors,
stabilizers, thermal traces and every diagonal two-point observable are tested.
Execute directly for a JSON audit, or discover the unittest class normally.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import scipy.linalg

import exact_thermal as target

ROOT = target.OUTPUT.parent
REFERENCE = ROOT / '3x3.json'
FIELDS = (
    "energy", "energy_density", "energy_variance", "entropy_nats",
    "entropy_density_nats", "free_energy", "free_energy_density",
    "dimensionless_free_energy_density", "log_partition",
    "magnetization_second_moment", "specific_heat_per_spin",
    "susceptibility_per_spin",
)
ATOL = 2e-10
AUDIT: dict = {}


def assert_close(actual, expected, **kwargs):
    """Require finite values and numerical agreement under supplied tolerances."""
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError("Nonfinite value in numerical correctness check")
    np.testing.assert_allclose(actual, expected, equal_nan=False, **kwargs)


def kron_operator(n: int, operators: dict[int, np.ndarray]) -> np.ndarray:
    """Build an operator with site zero as the rightmost Kronecker factor.

    Unspecified sites receive identity factors; row-major site zero is bit zero.
    """
    product = np.ones((1, 1), dtype=float)
    for site in reversed(range(n)):
        product = np.kron(product, operators.get(site, np.eye(2)))
    return product


def independent_dense(nx: int, ny: int, coupling: float = 1.,
                      field: float = .5) -> tuple[np.ndarray, np.ndarray]:
    """Build an independent open Pauli TFIM using full Kronecker products.

    Returns:
        The dense Hamiltonian and computational-basis observable diagonals,
        ordered as Mz squared followed by all ordered ZiZj pairs.
    """
    n = nx * ny
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    hamiltonian = np.zeros((2**n, 2**n))
    for y in range(ny):
        for x_site in range(nx):
            i = y * nx + x_site
            hamiltonian -= field * kron_operator(n, {i: x})
            if x_site + 1 < nx:
                hamiltonian -= coupling * kron_operator(n, {i: z, i + 1: z})
            if y + 1 < ny:
                hamiltonian -= coupling * kron_operator(n, {i: z, i + nx: z})
    z_diagonals = np.stack([kron_operator(n, {i: z}).diagonal() for i in range(n)])
    diagonal_observables = np.concatenate([
        (z_diagonals.sum(axis=0)**2)[None, :],
        np.einsum("ia,ja->ija", z_diagonals, z_diagonals).reshape(n*n, -1),
    ])
    return hamiltonian, diagonal_observables


def independent_permutation(nx: int, ny: int, x_reflect: bool,
                            y_reflect: bool, spin_flip: bool) -> np.ndarray:
    """Map basis labels under an independently constructed symmetry action."""
    n = nx * ny
    basis = np.arange(2**n, dtype=np.int64)
    transformed = np.zeros_like(basis)
    for row in range(ny):
        for col in range(nx):
            i = row * nx + col
            j = (ny - 1 - row if y_reflect else row) * nx
            j += nx - 1 - col if x_reflect else col
            transformed |= ((basis >> i) & 1) << j
    if spin_flip:
        transformed ^= 2**n - 1
    return transformed


def character_projector(nx: int, ny: int, signs: tuple[int, int, int]) -> np.ndarray:
    """Construct a dense character projector for two reflections and spin flip."""
    dim = 2**(nx*ny)
    projector = np.zeros((dim, dim))
    for actions in itertools.product((False, True), repeat=3):
        permutation = independent_permutation(nx, ny, *actions)
        character = np.prod([signs[i] if actions[i] else 1 for i in range(3)])
        projector[permutation, np.arange(dim)] += character / 8.
    return projector


def independent_thermal(energies: np.ndarray, vectors: np.ndarray,
                        diagonal_observables: np.ndarray,
                        beta: float, n: int) -> dict:
    """Evaluate independent Gibbs traces through full-basis populations.

    Assume a complete ascending real eigensystem, physical beta, and observable
    rows ordered as Mz squared then all ZiZj. Entropy uses natural logarithms;
    susceptibility is the spin-symmetric equal-time fluctuation beta <Mz^2>/n.
    """
    unnormalized = np.exp(-beta * (energies - energies[0]))
    probabilities = unnormalized / unnormalized.sum()
    energy = float(probabilities @ energies)
    variance = float(probabilities @ ((energies - energy)**2))
    logz = float(np.log(unnormalized.sum()) - beta * energies[0])
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    # Thermal computational-basis populations give diagonal observables without
    # using the solver's eigenstate-observable or sector projection machinery.
    state_probabilities = (vectors * vectors) @ probabilities
    observables = diagonal_observables @ state_probabilities
    return {
        "beta": beta,
        "energy": energy,
        "energy_density": energy / n,
        "energy_variance": variance,
        "entropy_nats": entropy,
        "entropy_density_nats": entropy / n,
        "free_energy": -logz / beta if beta else None,
        "free_energy_density": -logz / (beta*n) if beta else None,
        "dimensionless_free_energy_density": -logz/n,
        "log_partition": logz,
        "magnetization_second_moment": float(observables[0]),
        "specific_heat_per_spin": beta*beta*variance/n,
        "susceptibility_per_spin": beta*float(observables[0])/n,
        "correlations": observables[1:].reshape(n, n),
    }


class ExactThermalValidation(unittest.TestCase):
    """Independent small-square checks of sectors, spectra, and thermal traces."""

    @classmethod
    def setUpClass(cls):
        """Cache independent complete 2x2 and 3x3 dense eigensystems."""
        cls.cache = {}
        for nx, ny in ((2, 2), (3, 3)):
            hamiltonian, diagonal = independent_dense(nx, ny)
            energies, vectors = scipy.linalg.eigh(hamiltonian, driver="evd")
            cls.cache[nx, ny] = (hamiltonian, diagonal, energies, vectors)

    def test_01_projectors_signs_stabilizers_and_hamiltonians(self):
        """Check every character projector, including short-orbit stabilizers."""
        reports = []
        for (nx, ny), (h, diagonal, _, _) in self.cache.items():
            resolution = np.zeros_like(h)
            all_signs = list(target.sectors(nx, ny))
            self.assertEqual(set(map(tuple, all_signs)), set(itertools.product((-1, 1), repeat=3)))
            dims = []
            for signs in all_signs:
                block = target.build_sector(nx, ny, tuple(signs))
                u = block["projection"].toarray()
                reduced = block["H"].toarray()
                projector = character_projector(nx, ny, tuple(signs))
                expected_dim = int(round(float(np.trace(projector))))
                self.assertEqual(u.shape, (len(h), expected_dim))
                self.assertEqual(reduced.shape, (expected_dim, expected_dim))
                assert_close(u.T @ u, np.eye(expected_dim), atol=ATOL, rtol=0)
                assert_close(u @ u.T, projector, atol=ATOL, rtol=0)
                assert_close(reduced, u.T @ h @ u, atol=ATOL, rtol=0)
                assert_close(block["diagonal_observables"], diagonal @ (u*u), atol=ATOL, rtol=0)
                for generator in range(3):
                    actions = [False, False, False]
                    actions[generator] = True
                    permutation = independent_permutation(nx, ny, *actions)
                    assert_close(u[permutation], signs[generator]*u, atol=ATOL, rtol=0)
                resolution += u @ u.T
                dims.append(expected_dim)
            assert_close(resolution, np.eye(len(h)), atol=ATOL, rtol=0)
            self.assertEqual(sum(dims), len(h))
            # Distinct stabilizers give nonuniform block dimensions; treating
            # every orbit as length eight would fail the checks above.
            self.assertGreater(len(set(dims)), 1)
            reports.append({"system": f"{nx}x{ny}", "dimensions": dims,
                            "maximum_resolution_error": float(np.max(abs(resolution-np.eye(len(h)))))})
        AUDIT["projector_resolution_and_character_tests"] = reports

    def test_02_complete_spectra_and_trace_invariants(self):
        """Compare all-sector spectra and trace invariants with independent ED."""
        self.__class__.resolved = {}
        reports = []
        for (nx, ny), (h, _, dense_energies, _) in self.cache.items():
            solved = [target.solve_sector(nx, ny, tuple(signs), threads=target.THREADS)
                      for signs in target.sectors(nx, ny)]
            energies = np.concatenate([block["eigenvalues"] for block in solved])
            weights = np.concatenate([block["observable_weights"] for block in solved], axis=1)
            order = np.argsort(energies)
            energies, weights = energies[order], weights[:, order]
            self.__class__.resolved[nx, ny] = energies, weights
            assert_close(energies, dense_energies, atol=ATOL, rtol=0)
            n = nx*ny
            edges = nx*(ny-1)+ny*(nx-1)
            self.assertAlmostEqual(float(energies.mean()), 0., delta=ATOL)
            self.assertAlmostEqual(float(energies @ energies / len(energies)), edges + .25*n, delta=ATOL)
            self.assertAlmostEqual(float(np.trace(h)), 0., delta=ATOL)
            # A degenerate spectrum must be allowed: tests compare invariant
            # thermal traces, never eigenvectors across degenerate subspaces.
            degeneracies = int(np.sum(np.diff(dense_energies) < 1e-10))
            self.assertGreater(degeneracies, 0)
            reports.append({"system": f"{nx}x{ny}", "spectrum_error": float(np.max(abs(energies-dense_energies))),
                            "trace_h2_per_state": float(energies @ energies / len(energies)),
                            "adjacent_degeneracies": degeneracies})
        AUDIT["complete_spectrum_and_trace_tests"] = reports

    def test_03_thermal_traces_and_all_pair_correlations(self):
        """Compare Gibbs scalars and every Pauli pair correlation across betas."""
        reports = []
        for (nx, ny), (_, diagonal, dense_energies, vectors) in self.cache.items():
            if (nx, ny) not in getattr(self.__class__, "resolved", {}):
                self.test_02_complete_spectra_and_trace_invariants()
            energies, weights = self.resolved[nx, ny]
            betas = (0., .05, .2, .4, .7, 1., 2.)
            rows = target.thermal_rows(energies, weights, betas, nx*ny)
            self.assertEqual(len(rows), len(betas))
            error = 0.
            for beta, actual in zip(betas, rows):
                expected = independent_thermal(dense_energies, vectors, diagonal, beta, nx*ny)
                for key in FIELDS:
                    if expected[key] is None:
                        self.assertIsNone(actual[key], f"{key} at beta=0 must not emit JSON Inf/NaN")
                    else:
                        assert_close(actual[key], expected[key], atol=ATOL, rtol=1e-11, err_msg=key)
                        error = max(error, abs(actual[key]-expected[key]))
                correlations = np.asarray(actual["longitudinal_correlations"])
                assert_close(correlations, expected["correlations"], atol=ATOL, rtol=0)
                assert_close(correlations, correlations.T, atol=ATOL, rtol=0)
                assert_close(correlations.diagonal(), 1., atol=ATOL, rtol=0)
                self.assertGreaterEqual(np.linalg.eigvalsh(correlations).min(), -ATOL)
                self.assertAlmostEqual(float(correlations.sum()), actual["magnetization_second_moment"], delta=ATOL)
                self.assertAlmostEqual(actual["dimensionless_free_energy_density"],
                                       (beta*actual["energy"]-actual["entropy_nats"])/(nx*ny), delta=ATOL)
                self.assertGreaterEqual(actual["entropy_nats"], -ATOL)
                self.assertLessEqual(actual["entropy_nats"], nx*ny*np.log(2)+ATOL)
            zero = rows[0]
            assert_close(zero["longitudinal_correlations"], np.eye(nx*ny), atol=ATOL, rtol=0)
            self.assertAlmostEqual(zero["entropy_nats"], nx*ny*np.log(2), delta=ATOL)
            self.assertEqual(zero["specific_heat_per_spin"], 0.)
            self.assertEqual(zero["susceptibility_per_spin"], 0.)
            reports.append({"system": f"{nx}x{ny}", "beta_grid": list(betas), "max_scalar_error": error})
        AUDIT["thermal_and_all_pair_tests"] = reports

    def test_04_saved_exact_3x3_reference(self):
        """Check the saved 3x3 digest, source binding, and thermal scalar fields."""
        payload_bytes = REFERENCE.read_bytes()
        expected_hash = REFERENCE.with_suffix(".json.sha256").read_text().strip().split()[0]
        digest = hashlib.sha256(payload_bytes).hexdigest()
        self.assertEqual(digest, expected_hash)
        saved = json.loads(payload_bytes)
        if (3, 3) not in getattr(self.__class__, "resolved", {}):
            self.test_02_complete_spectra_and_trace_invariants()
        energies, weights = self.resolved[3, 3]
        if (saved.get('system') != '3x3' or saved.get('passed') is not True
                or saved.get('source_sha256') != target.source_hashes()):
            raise AssertionError('fresh 3x3 input contract differs')
        recomputed = target.thermal_rows(energies, weights, [r["beta"] for r in saved["thermal"]], 9)
        max_error = 0.
        for actual, reference in zip(recomputed, saved["thermal"]):
            for key in FIELDS:
                assert_close(actual[key], reference[key], atol=ATOL, rtol=1e-11, err_msg=key)
                max_error = max(max_error, abs(actual[key]-reference[key]))
        AUDIT["saved_3x3_reference"] = {"path": str(REFERENCE.relative_to(ROOT)),
                                        "sha256": digest, "max_error": max_error,
                                        "rows": len(saved["thermal"])}

    def test_05_commuting_and_zero_hamiltonian_degeneracies(self):
        """Check commuting and zero-Hamiltonian limits without fixing eigenbases."""
        reports = []
        for coupling, field in ((1., 0.), (0., .5), (0., 0.)):
            h, diagonal = independent_dense(2, 2, coupling=coupling, field=field)
            energies, vectors = scipy.linalg.eigh(h, driver="evd")
            block_energies, block_weights = [], []
            for signs in target.sectors(2, 2):
                block = target.build_sector(2, 2, tuple(signs), coupling=coupling, field=field)
                u = block["projection"].toarray()
                assert_close(block["H"].toarray(), u.T @ h @ u, atol=ATOL, rtol=0)
                values, eigenvectors = scipy.linalg.eigh(block["H"].toarray(), driver="evd")
                block_energies.append(values)
                block_weights.append(block["diagonal_observables"] @ (eigenvectors*eigenvectors))
            combined = np.concatenate(block_energies)
            weights = np.concatenate(block_weights, axis=1)
            assert_close(np.sort(combined), energies, atol=ATOL, rtol=0)
            for beta in (0., .4, 3.):
                actual = target.thermal_rows(combined, weights, [beta], 4)[0]
                expected = independent_thermal(energies, vectors, diagonal, beta, 4)
                for key in FIELDS:
                    if expected[key] is not None:
                        assert_close(actual[key], expected[key], atol=ATOL, rtol=1e-11)
                assert_close(actual["longitudinal_correlations"], expected["correlations"], atol=ATOL, rtol=0)
            if field == 0 and coupling == 1:
                low = target.thermal_rows(combined, weights, [20.], 4)[0]
                self.assertAlmostEqual(low["entropy_nats"], np.log(2.), delta=ATOL)
                assert_close(low["longitudinal_correlations"], np.ones((4, 4)), atol=ATOL, rtol=0)
            reports.append({"coupling": coupling, "field": field, "status": "PASS"})
        AUDIT["commuting_and_fully_degenerate_limits"] = reports


def main() -> int:
    """Run the five validation tests and print a finite JSON audit.

    Returns:
        Zero if every test succeeds, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=target.THREADS)
    parser.parse_args()
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExactThermalValidation)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    warning_records = sorted({(str(w.message), str(w.category.__name__)) for w in captured})
    payload = {"status": "PASS" if result.wasSuccessful() else "FAIL",
               "source_sha256": {
                   Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in (__file__, target.__file__)
               },
               "configured_blas_threads": target.THREADS,
               "tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "wall_seconds": time.perf_counter()-start,
               "absolute_tolerance": ATOL,
               "scope": "Independent Kronecker dense 2x2/3x3 checks; no full 4x4 diagonalization",
               "runtime_warnings": [{"message": msg, "category": category} for msg, category in warning_records],
               "runtime_warning_count": len(captured),
               "finite_checks": "Every numerical array comparison explicitly rejects NaN/Inf",
               "checks": AUDIT}
    output = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n"
    print(output)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
