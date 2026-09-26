#!/usr/bin/env python3
"""Complete thermal ED of open TFIM squares in eight real symmetry sectors.

H=-J sum_<ij> ZiZj-h sum_i Xi; Pauli operators, row-major sites, physical
inverse temperature and entropy in nats. The only reductions are commuting
horizontal/vertical reflections and global spin inversion. ALL sectors are
required for thermal traces; no magnetization or translation constraint.

Internal solver module: the historical direct CLI is disabled. Invoke
  python references/run_references.py --data-root /external/output --threads 4
from the release root. That entry point sets the external output directory
and builds fresh independently validated inputs. No original data is read.
Fresh processes are required to configure BLAS/LAPACK threads.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from functools import lru_cache
import io
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time
import warnings

# Set before NumPy/SciPy load. Accelerate does not expose active-thread counts:
# this is a configured maximum, not a measurement of actual worker threads.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--threads', type=int, default=int(os.environ.get('VECLIB_MAXIMUM_THREADS', '4')))
_thread_args, _ = _pre.parse_known_args()
THREADS = _thread_args.threads
if not 1 <= THREADS <= 12:
    raise ValueError('threads must be between 1 and 12')
for _key in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = str(THREADS)

import numpy as np
import psutil
import scipy
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.special import logsumexp

if __package__:
    from .io_utils import immutable_bytes, immutable_json, read_verified, sha256
else:
    from io_utils import immutable_bytes, immutable_json, read_verified, sha256

HERE = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get('MPS_REFERENCE_DATA_ROOT', 'data')).resolve() / 'references/exact_4x4'
BETAS = (0.05, 0.2, 0.4, 0.45, 0.5, 0.7, 0.85, 2.0)
GROUND = OUTPUT.parent / 'ground_4x4.json'
SCHEMA = 1
DRIVER = 'evd'


def source_hashes():
    """Return SHA-256 digests of the six reference implementation modules."""
    return {name: sha256(HERE / name) for name in (
        'exact_thermal.py', 'validate_square.py', 'ground_state.py',
        'exact_chain.py', 'run_references.py', 'io_utils.py')}


def ground_reference():
    """Load and validate the source-bound independent 4x4 ground reference.

    Reject stale source hashes, different physics, absent validation, or
    failed numerical gates before returning the ground-state record.
    """
    # Bind freshly validated inputs, never a host-dependent historical digest.
    payload = read_verified(GROUND)
    if (payload.get('dataset_kind') != 'exact_ground_reference'
            or payload.get('system') != '4x4'
            or payload.get('passed') is not True
            or payload.get('source_sha256') != source_hashes()
            or payload.get('physics') != {'J': 1., 'h': .5, 'boundary': 'open'}
            or not payload.get('validation', {}).get('passed')):
        raise ValueError('fresh ground-state input contract/validation differs')
    if __package__:
        from .run_references import validate_ground
    else:
        from run_references import validate_ground
    validate_ground(payload['ground'], payload['solver_checks'])
    return payload['ground']


def sectors(nx, ny):
    """Enumerate all reflection and spin-inversion sign triples.

    Signs are ordered as horizontal reflection, vertical reflection, and
    global spin inversion. Some sectors can be empty for small rectangles.

    Raises:
        ValueError: The rectangle is empty or contains more than 16 sites.
    """
    if nx < 1 or ny < 1 or nx * ny > 16:
        raise ValueError('positive rectangle with at most 16 sites required')
    return tuple(itertools.product((1, -1), repeat=3))


def sector_name(signs):
    """Encode a valid reflection/reflection/spin-inversion sign triple."""
    if tuple(signs) not in sectors(4, 4):
        raise ValueError('three signs in {+1,-1} required')
    return '_'.join(label + ('p' if s == 1 else 'm')
                    for label, s in zip(('px', 'py', 'spin'), signs))


def edges(nx, ny):
    """Return open nearest-neighbor rectangle edges in row-major site order."""
    return [(y * nx + x, y * nx + x + 1)
            for y in range(ny) for x in range(nx - 1)] + [
            (y * nx + x, (y + 1) * nx + x)
            for y in range(ny - 1) for x in range(nx)]


@lru_cache(maxsize=4)
def full_system(nx, ny, coupling=1., field=.5):
    """Build the full sparse Pauli TFIM and site-resolved Z diagonals.

    Returns:
        The CSR Hamiltonian and an array shaped (num_spins, 2**num_spins).
        Site zero is the least-significant computational-basis bit.
    """
    sectors(nx, ny)
    n = nx * ny
    d = 1 << n
    states = np.arange(d, dtype=np.int64)
    z = np.asarray([1 - 2 * ((states >> i) & 1) for i in range(n)], dtype=np.float64)
    diag = -coupling * sum((z[i] * z[j] for i, j in edges(nx, ny)), np.zeros(d))
    rows = np.tile(states, n + 1)
    cols = np.concatenate([states] + [states ^ (1 << i) for i in range(n)])
    data = np.concatenate([diag, np.full(n * d, -field)])
    h = coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()
    h.eliminate_zeros()
    return h, z


@lru_cache(maxsize=4)
def group_images(nx, ny):
    """Map every basis label under the eight commuting symmetry actions.

    Returns:
        An eight-row array of transformed basis labels and the corresponding
        horizontal-reflection, vertical-reflection, spin-inversion powers.
    """
    sectors(nx, ny)
    n = nx * ny
    states = np.arange(1 << n, dtype=np.int64)
    maps = ([y * nx + (nx - 1 - x) for y in range(ny) for x in range(nx)],
            [(ny - 1 - y) * nx + x for y in range(ny) for x in range(nx)])
    images = []
    powers = tuple(itertools.product((0, 1), repeat=3))
    for a, b, c in powers:
        perm = np.arange(n)
        if a:
            perm = np.asarray(maps[0])[perm]
        if b:
            perm = np.asarray(maps[1])[perm]
        image = np.zeros_like(states)
        for source, target in enumerate(perm):
            image |= ((states >> source) & 1) << target
        if c:
            image ^= (1 << n) - 1
        images.append(image)
    return np.asarray(images), powers


def build_sector(nx, ny, signs, coupling=1., field=.5):
    """Project signed symmetry orbits with exact stabilizer selection rules.

    Every column has disjoint orbit support. For ANY Z-diagonal observable,
    Q.T O Q is diagonal and equals its orbit average, even when O itself does
    not commute with the spatial reflections (e.g. an individual ZiZj).

    Returns:
        A mapping containing the reduced CSR Hamiltonian, orthonormal sparse
        projection, and diagonal observable rows ordered as Mz squared then
        every row-major ordered pair ZiZj.
    """
    sector_name(signs)
    images, powers = group_images(nx, ny)
    reps = np.unique(images.min(axis=0))
    characters = np.asarray([np.prod([s ** k for s, k in zip(signs, power)])
                             for power in powers], dtype=np.float64)
    q = coo_matrix((np.repeat(characters, reps.size),
                    (images[:, reps].ravel(), np.tile(np.arange(reps.size), 8))),
                   shape=(1 << (nx * ny), reps.size)).tocsr()
    q.sum_duplicates()
    q.eliminate_zeros()
    norm2 = np.asarray(q.power(2).sum(axis=0)).ravel()
    keep = norm2 > 0
    q = q[:, keep].multiply(1. / np.sqrt(norm2[keep])).tocsr()
    full_h, z = full_system(nx, ny, coupling, field)
    h = (q.T @ full_h @ q).tocsr()
    h.eliminate_zeros()
    delta = h - h.T
    if delta.nnz and np.max(np.abs(delta.data)) > 2e-14:
        raise AssertionError('reduced Hamiltonian is not symmetric')
    prob_t = q.power(2).T.tocsr()
    n = nx * ny
    diagobs = np.empty((1 + n * n, h.shape[0]), dtype=np.float64)
    diagobs[0] = prob_t @ np.square(z.sum(axis=0))
    for i in range(n):
        for j in range(n):
            diagobs[1 + i * n + j] = prob_t @ (z[i] * z[j])
    return {'H': h, 'projection': q, 'diagonal_observables': diagobs}


def peak_rss_bytes():
    """Return process peak resident memory in bytes across supported platforms."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else 1024 * value)


def solve_sector(nx, ny, signs, threads=THREADS, progress=False):
    """Diagonalize one complete symmetry sector and audit observable weights.

    Use the module's default J=1 and h=0.5. Residual and orthogonality checks
    sample eigenvectors; spectral and observable trace checks use all levels.

    Returns:
        Eigenvalues, Mz-squared/all-pair observable weights, audit metrics,
        and elapsed timings. Empty sectors return empty arrays.

    Raises:
        ValueError: Requested threads differ from the import-time setting.
        MemoryError: Available memory fails the conservative allocation gate.
        AssertionError: A numerical eigensystem or observable audit fails.
    """
    if threads != THREADS:
        raise ValueError('thread configuration requires a fresh CLI process')
    begin = time.perf_counter()
    block = build_sector(nx, ny, signs)
    h = block['H']
    size = h.shape[0]
    build_seconds = time.perf_counter() - begin
    if not size:
        return {'eigenvalues': np.empty(0),
                'observable_weights': np.empty((1 + (nx * ny) ** 2, 0)),
                'timings': {'build_seconds': build_seconds, 'eigh_seconds': 0.,
                            'weights_seconds': 0., 'audit_seconds': 0.,
                            'solve_seconds': time.perf_counter() - begin}, 'checks': {}}
    # evd: input/eigenvectors, O(2d²) LAPACK workspace, allocator overhead.
    # Conservative available-memory gate, not a strict claim about peak RSS.
    estimated_extra = 5 * 8 * size * size + (256 << 20)
    available = psutil.virtual_memory().available
    if available < estimated_extra + (512 << 20):
        raise MemoryError(f'need {estimated_extra / 2**30:.2f} GiB plus reserve; '
                          f'available {available / 2**30:.2f} GiB')
    if progress:
        print(f'{sector_name(signs)}: d={size}; diagonalizing, {threads} configured threads', flush=True)
    start = time.perf_counter()
    dense = h.toarray(order='F')
    values, vectors = eigh(dense, overwrite_a=True, check_finite=False, driver=DRIVER)
    del dense
    eigh_seconds = time.perf_counter() - start
    start = time.perf_counter()
    weights = np.empty((1 + (nx * ny) ** 2, size), dtype=np.float64)
    for first in range(0, size, 128):
        last = min(first + 128, size)
        weights[:, first:last] = block['diagonal_observables'] @ np.square(vectors[:, first:last])
    weights_seconds = time.perf_counter() - start
    start = time.perf_counter()
    indices = np.unique(np.r_[np.arange(min(3, size)), np.linspace(0, size - 1, 20, dtype=int)])
    selected = vectors[:, indices]
    residual = h @ selected - selected * values[indices]
    max_residual = float(np.max(np.linalg.norm(residual, axis=0)))
    gram_error = float(np.max(np.abs(selected.T @ selected - np.eye(len(indices)))))
    trace_error = abs(float(values.sum()) - float(h.diagonal().sum()))
    trace2_error = abs(float(values @ values) - float(h.multiply(h).sum()))
    obs_trace_error = float(np.max(np.abs(weights.sum(axis=1) - block['diagonal_observables'].sum(axis=1))))
    diagonal_corr_error = float(np.max(np.abs(weights[1:].reshape(nx*ny, nx*ny, size)[np.arange(nx*ny), np.arange(nx*ny)] - 1.)))
    if not np.isfinite([max_residual, gram_error, trace_error, trace2_error,
                        obs_trace_error, diagonal_corr_error]).all():
        raise AssertionError('nonfinite eigensystem audit metric')
    if (max_residual > 2e-10 or gram_error > 2e-11 or trace_error > 2e-8
            or trace2_error > 1e-9 * max(1., float(values @ values))
            or obs_trace_error > 2e-7 or diagonal_corr_error > 2e-11):
        raise AssertionError('eigensystem/observable audit failed')
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise AssertionError('nonfinite eigensystem')
    checks = {'sampled_eigenpair_residual_max_l2': max_residual,
              'sampled_orthogonality_max_abs': gram_error,
              'trace_error': trace_error, 'trace_h2_error': trace2_error,
              'observable_trace_max_abs_error': obs_trace_error,
              'diagonal_correlation_max_abs_error': diagonal_corr_error,
              'eigenpairs_checked': indices.tolist()}
    return {'eigenvalues': values, 'observable_weights': weights, 'checks': checks,
            'timings': {'build_seconds': build_seconds, 'eigh_seconds': eigh_seconds,
                        'weights_seconds': weights_seconds,
                        'audit_seconds': time.perf_counter() - start,
                        'solve_seconds': time.perf_counter() - begin}}


def thermal_rows(eigenvalues, obs_weights, betas, num_spins):
    """Evaluate Gibbs traces from a complete spectrum and observable weights.

    Args:
        eigenvalues: All 2**num_spins energies, combined across every sector.
        obs_weights: Eigenstate expectations, with Mz squared first followed
            by all ordered ZiZj pairs; columns correspond to eigenvalues.
        betas: Finite nonnegative physical inverse temperatures.
        num_spins: Number of Pauli spins.

    Returns:
        JSON-ready rows with entropy in nats and equal-time fluctuation
        susceptibility beta <Mz squared>/N, not the Kubo response. Dimensional
        free energies are None at beta zero; dimensionless free energy remains
        defined. The zero mean Mz assumes the full spin-inversion-symmetric trace.

    Raises:
        ValueError: Array sizes, finite values, or beta values are invalid.
    """
    e = np.asarray(eigenvalues, dtype=float)
    o = np.asarray(obs_weights, dtype=float)
    if e.shape != (1 << num_spins,) or o.shape != (1 + num_spins**2, len(e)):
        raise ValueError('thermal trace requires the COMPLETE spectrum and all observable weights')
    if not np.isfinite(e).all() or not np.isfinite(o).all():
        raise ValueError('nonfinite spectrum or observables')
    rows = []
    for beta in betas:
        beta = float(beta)
        if not np.isfinite(beta) or beta < 0:
            raise ValueError('finite nonnegative physical beta required')
        logw = -beta * (e - e.min())
        logz_shifted = float(logsumexp(logw))
        w = np.exp(logw - logz_shifted)
        logz = logz_shifted - beta * float(e.min())
        energy = float(w @ e)
        variance = float(w @ np.square(e - energy))
        measured = o @ w
        entropy = float(-w @ np.log(np.maximum(w, np.finfo(float).tiny)))
        rows.append({'num_spins': num_spins, 'beta': beta, 'energy': energy,
                     'energy_density': energy/num_spins, 'energy_variance': variance,
                     'specific_heat_per_spin': beta**2 * variance/num_spins,
                     'log_partition': logz, 'free_energy': -logz/beta if beta else None,
                     'free_energy_density': -logz/(beta*num_spins) if beta else None,
                     'entropy_nats': entropy, 'entropy_density_nats': entropy/num_spins,
                     'dimensionless_free_energy_density': -logz/num_spins,
                     'mean_longitudinal_magnetization': 0.,
                     'magnetization_second_moment': float(measured[0]),
                     'susceptibility_per_spin': beta*float(measured[0])/num_spins,
                     'longitudinal_correlations': measured[1:].reshape(num_spins, num_spins).tolist(),
                     'thermodynamic_identity_residual': beta*energy - entropy + logz})
    return rows


def environment():
    """Record numerical versions, hardware, and configured thread limits.

    Configured thread limits are not measurements of active BLAS workers.
    """
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.show_config()
    return {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
            'platform': platform.platform(), 'machine': platform.machine(),
            'logical_cpus': psutil.cpu_count() or os.cpu_count(), 'physical_cpus': psutil.cpu_count(logical=False),
            'total_memory_bytes': psutil.virtual_memory().total,
            'configured_max_threads': THREADS,
            'actual_blas_threads': 'unavailable for Apple Accelerate; configured maximum only',
            'numpy_build': stream.getvalue()}


def contract():
    """Return the source-bound 4x4 physics, solver, and observable contract."""
    return {'schema_version': SCHEMA, 'source_hashes_sha256': source_hashes(),
            'nx': 4, 'ny': 4, 'coupling_J': 1., 'transverse_field_h': .5,
            'boundary': 'open', 'ordering': 'row-major; site0=least-significant-bit',
            'entropy_logarithm': 'natural', 'beta': 'physical inverse temperature',
            'symmetries': ['horizontal reflection', 'vertical reflection', 'global spin inversion'],
            'all_sectors_required': True, 'dtype': 'float64', 'driver': DRIVER,
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'observable_order': 'Mz²; ZiZj for every ordered pair i,j in row-major order',
            'chi_definition': 'beta*(<Mz²>-<Mz>²)/N; equal-time fluctuation, NOT Kubo susceptibility',
            'cv_definition': 'beta²*(<H²>-<H>²)/N'}


def low_spectrum_validation():
    """Check all 4x4 sectors with sparse low levels and trace identities.

    Compare parity-resolved levels and even ground observables against the
    independent ground reference without a complete dense diagonalization.
    """
    ground = ground_reference()
    rows = []
    trace = trace2 = 0.
    obs_trace = np.zeros(257)
    dimension = 0
    for signs in sectors(4, 4):
        block = build_sector(4, 4, signs)
        h = block['H']
        d = h.shape[0]
        values, vectors = eigsh(h, k=3, which='SA', tol=1e-13, maxiter=10000,
                                ncv=48, v0=np.sin(np.arange(d) + .37))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise AssertionError('nonfinite 4x4 sparse eigensystem')
        residual = float(np.max(np.linalg.norm(h @ vectors - vectors * values, axis=0)))
        if not np.isfinite(residual) or residual > 1e-9:
            raise AssertionError('4x4 sparse residual failed')
        row = {'signs': list(signs), 'dimension': d, 'lowest_three': values.tolist(),
               'max_eigenpair_residual_l2': residual}
        if signs == (1, 1, 1):
            weights = block['diagonal_observables'] @ np.square(vectors[:, 0])
            if not np.isfinite(weights).all():
                raise AssertionError('nonfinite 4x4 sparse observable weights')
            row['ground_energy_error'] = abs(float(values[0]) - ground['energy'])
            row['ground_mz2_error'] = abs(float(weights[0]) - ground['magnetization_second_moment'])
            row['ground_correlations_max_abs_error'] = float(np.max(np.abs(
                weights[1:].reshape(16, 16) - ground['longitudinal_correlations'])))
            if (row['ground_energy_error'] > 2e-10 or row['ground_mz2_error'] > 2e-8
                    or row['ground_correlations_max_abs_error'] > 2e-9):
                raise AssertionError('saved 4x4 ground-state cross-check failed')
        dimension += d
        trace += float(h.diagonal().sum())
        trace2 += float(h.multiply(h).sum())
        obs_trace += block['diagonal_observables'].sum(axis=1)
        rows.append(row)
    target_dimensions = [8384, 8192, 8128, 8192, 8128, 8192, 8128, 8192]
    if [r['dimension'] for r in rows] != target_dimensions or dimension != 65536:
        raise AssertionError('4x4 sector dimension accounting failed')
    parity_errors = {}
    for parity in (1, -1):
        computed = sorted(v for r in rows if r['signs'][2] == parity for v in r['lowest_three'])[:3]
        saved = sorted(r['energy'] for r in ground['low_lying_levels'] if r['parity'] == parity)
        err = float(np.max(np.abs(np.asarray(computed) - saved)))
        if err > 2e-10:
            raise AssertionError('saved parity-resolved low levels differ')
        parity_errors[str(parity)] = err
    expected_obs = np.r_[16., np.eye(16).ravel()]
    if (abs(trace) > 2e-8 or abs(trace2 / dimension - 28.) > 2e-11
            or np.max(np.abs(obs_trace / dimension - expected_obs)) > 2e-11):
        raise AssertionError('4x4 full-space trace/infinite-temperature identity failed')
    return {'passed': True, 'sectors': rows, 'dimension': dimension,
            'trace_H': trace, 'trace_H2_per_state': trace2 / dimension,
            'infinite_temperature_observables_max_abs_error': float(np.max(np.abs(
                obs_trace / dimension - expected_obs))),
            'saved_low_levels_max_abs_errors_by_parity': parity_errors,
            'ground_reference_sha256': sha256(GROUND)}


def validate(output):
    """Verify or publish source-bound small-system and sparse 4x4 validation.

    Run the independent validator in a subprocess, require all five tests,
    and commit the report without replacing different existing output.
    """
    path = output / 'validation.json'
    if path.exists():
        verify_validation(output)
        print(f'Already validated current code: {path}', flush=True)
        return
    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(HERE / 'validate_square.py'),
                             '--threads', str(THREADS)],
                            capture_output=True, text=True, env=os.environ.copy())
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError('independent small-system tests failed; refusing production')
    independent = json.loads(result.stdout)
    if (independent['status'] != 'PASS' or independent['tests_run'] != 5
            or independent['failures'] or independent['errors']):
        raise AssertionError('independent validation coverage/status mismatch')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', RuntimeWarning)
        sparse = low_spectrum_validation()
    payload = {'passed': True, 'contract': contract(), 'environment': environment(),
               'independent_validation': independent, 'independent_test_stderr': result.stderr,
               'sparse_validation_runtime_warnings': dict(Counter(str(w.message) for w in captured)),
               '4x4_sparse_validation': sparse, 'wall_seconds': time.perf_counter() - start}
    immutable_json(path, payload)
    print(f'PASS validation saved: {path}', flush=True)


def verify_validation(output):
    """Verify the saved validation digest, current contract, and ground binding."""
    validation = read_verified(output / 'validation.json')
    if validation.get('passed') is not True or validation.get('contract') != contract():
        raise ValueError('missing/current validation contract differs; validate in a fresh output directory')
    if validation['4x4_sparse_validation']['ground_reference_sha256'] != sha256(GROUND):
        raise ValueError('saved independent ground reference changed')
    return validation


def swap_snapshot():
    """Return system-wide swap statistics or an explicit unavailable status."""
    try:
        return {'status': 'available', **psutil.swap_memory()._asdict()}
    except (PermissionError, OSError, psutil.Error) as exc:
        return {'status': 'unavailable', 'reason': str(exc)}


class Monitor:
    """Lightweight RSS/memory sampling and visible liveness during LAPACK."""
    def __init__(self):
        """Initialize memory samples and create an unstarted daemon poller."""
        self.done = threading.Event()
        self.start = time.perf_counter()
        self.rss = psutil.Process().memory_info().rss
        self.min_available = psutil.virtual_memory().available
        self.swap_before = swap_snapshot()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        """Sample memory until stopped and print periodic liveness updates."""
        last_print = self.start
        while not self.done.wait(.2):
            self.rss = max(self.rss, psutil.Process().memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            now = time.perf_counter()
            if now - last_print >= 15:
                print(f'  still working: {now-self.start:.0f}s elapsed; '
                      f'peak RSS {self.rss/2**30:.2f} GiB; '
                      f'available {psutil.virtual_memory().available/2**30:.2f} GiB', flush=True)
                last_print = now

    def finish(self):
        """Stop and join the started poller, then return memory diagnostics.

        Swap statistics are system-wide and not attributed to this process.
        """
        self.done.set()
        self.thread.join()
        return {'sampled_peak_rss_bytes': self.rss, 'process_peak_rss_bytes': peak_rss_bytes(),
                'minimum_system_available_bytes': self.min_available,
                'system_swap_before': self.swap_before, 'system_swap_after': swap_snapshot(),
                'swap_scope': 'system-wide, not attributable solely to this process'}


def checkpoint_paths(output, signs):
    """Return the NPZ data and JSON metadata paths for a symmetry sector."""
    base = output / 'sectors' / sector_name(signs)
    return base.with_suffix('.npz'), base.with_suffix('.json')


def load_checkpoint(output, signs):
    """Load a complete checkpoint after digest, contract, and shape checks.

    Returns:
        Eigenvalues, observable weights, and metadata, or None if neither
        checkpoint file exists. The saved passed flag is required; numerical
        audit metrics are not recomputed.

    Raises:
        ValueError: The checkpoint is incomplete, stale, or malformed.
    """
    data_path, meta_path = checkpoint_paths(output, signs)
    if not data_path.exists() and not meta_path.exists():
        return None
    if not data_path.exists() or not meta_path.exists():
        raise ValueError(f'incomplete checkpoint; preserve and investigate: {data_path}')
    meta = read_verified(meta_path)
    if (meta['contract'] != contract() or meta['signs'] != list(signs)
            or meta['data_sha256'] != sha256(data_path) or meta.get('passed') is not True
            or meta['validation_sha256'] != sha256(output / 'validation.json')):
        raise ValueError(f'checkpoint binding/audit differs: {meta_path}')
    with np.load(data_path, allow_pickle=False) as f:
        values, weights = f['eigenvalues'], f['observable_weights']
    d = meta['dimension']
    expected_dimensions = dict(zip(sectors(4,4), (8384,8192,8128,8192,8128,8192,8128,8192)))
    if d != expected_dimensions[tuple(signs)]:
        raise ValueError('checkpoint symmetry-sector dimension differs')
    if (values.shape != (d,) or weights.shape != (257, d)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(np.diff(values) < 0)):
        raise ValueError('malformed sector checkpoint')
    return values, weights, meta


def compute_checkpoint(output, signs):
    """Reuse a verified sector or solve, audit, and publish it immutably.

    Return eigenvalues, observable weights, and metadata. Reuse retains
    historical timings; a fresh solve records memory and warning diagnostics
    and checks the ground reference for the all-even sector.
    """
    prior = load_checkpoint(output, signs)
    if prior is not None:
        print(f'Reusing verified {sector_name(signs)}, d={len(prior[0])}; original timings '
              f'used {prior[2]["environment"]["configured_max_threads"]} configured threads '
              '(no new timing measurement)', flush=True)
        return prior
    start = time.perf_counter()
    monitor = Monitor()
    monitor.thread.start()
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always', RuntimeWarning)
            result = solve_sector(4, 4, signs, progress=True)
        memory = monitor.finish()
    except BaseException:
        monitor.finish()
        raise
    checks = result['checks']
    weights = result['observable_weights']
    sum_error = float(np.max(np.abs(weights[1:].sum(axis=0) - weights[0])))
    if not np.isfinite(sum_error) or sum_error > 2e-9 or weights[0].min() < -2e-10 or weights[0].max() > 256+2e-9:
        raise AssertionError('magnetization/correlation sum rule failed')
    checks['magnetization_correlation_sum_max_abs_error'] = sum_error
    if signs == (1, 1, 1):
        ground = ground_reference()
        checks['saved_ground_energy_abs_error'] = abs(result['eigenvalues'][0] - ground['energy'])
        checks['saved_ground_mz2_abs_error'] = abs(weights[0, 0] - ground['magnetization_second_moment'])
        checks['saved_ground_correlations_max_abs_error'] = float(np.max(np.abs(
            weights[1:, 0].reshape(16, 16) - ground['longitudinal_correlations'])))
        if (checks['saved_ground_energy_abs_error'] > 2e-10
                or checks['saved_ground_mz2_abs_error'] > 2e-8
                or checks['saved_ground_correlations_max_abs_error'] > 2e-9):
            raise AssertionError('complete ED disagrees with independent ground-state reference')
    data_path, meta_path = checkpoint_paths(output, signs)
    save_start = time.perf_counter()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, eigenvalues=result['eigenvalues'], observable_weights=weights)
    immutable_bytes(data_path, buffer.getvalue())
    save_seconds = time.perf_counter() - save_start
    counts = Counter(str(w.message) for w in recorded)
    meta = {'passed': True, 'contract': contract(), 'validation_sha256': sha256(output / 'validation.json'),
            'signs': list(signs), 'dimension': len(result['eigenvalues']),
            'environment': environment(), 'checks': checks, 'memory': memory,
            'runtime_warnings': dict(counts),
            'timings': {**result['timings'], 'checkpoint_write_seconds': save_seconds,
                        'compute_and_data_write_seconds': time.perf_counter() - start},
            'data_sha256': sha256(data_path), 'data_bytes': data_path.stat().st_size}
    immutable_json(meta_path, meta)
    print(f'Saved {sector_name(signs)}: {meta["timings"]["compute_and_data_write_seconds"]:.2f}s, '
          f'peak RSS {memory["process_peak_rss_bytes"]/2**30:.2f} GiB', flush=True)
    return result['eigenvalues'], weights, meta


def run_all(output):
    """Assemble all eight validated sectors and publish 4x4 thermal traces.

    Verify complete-space, infinite-temperature, thermodynamic, and
    correlation identities before writing the immutable thermal JSON.
    """
    verify_validation(output)
    values, weights = [], []
    for i, signs in enumerate(sectors(4, 4), 1):
        print(f'[{"="*(i-1)}>{"."*(8-i)}] sector {i}/8', flush=True)
        v, w, _ = compute_checkpoint(output, signs)
        values.append(v)
        weights.append(w)
    all_values = np.concatenate(values)
    all_weights = np.concatenate(weights, axis=1)
    rows = thermal_rows(all_values, all_weights, (0.,) + BETAS, 16)
    zero = rows[0]
    if (abs(all_values.sum()) > 2e-8 or abs(all_values@all_values/65536-28.) > 2e-10
            or abs(zero['energy']) > 2e-11 or abs(zero['entropy_nats']-16*np.log(2)) > 2e-11
            or abs(zero['magnetization_second_moment']-16) > 2e-10
            or np.max(np.abs(np.asarray(zero['longitudinal_correlations'])-np.eye(16))) > 2e-11
            or max(abs(r['thermodynamic_identity_residual']) for r in rows) > 2e-10):
        raise AssertionError('complete thermal spectrum validation failed')
    for row in rows:
        corr = np.asarray(row['longitudinal_correlations'])
        if (abs(corr.sum()-row['magnetization_second_moment']) > 2e-8
                or np.max(np.abs(corr-corr.T)) > 2e-11
                or np.linalg.eigvalsh(corr).min() < -2e-10):
            raise AssertionError('complete thermal correlation check failed')
    payload = {'schema_version': SCHEMA, 'dataset_kind': 'exact_symmetry_resolved_thermal_reference',
               'contract': contract(), 'method': 'complete real float64 ED of all eight symmetry sectors',
               'system': {'geometry': 'open_square', 'nx': 4, 'ny': 4, 'num_spins': 16,
                          'hilbert_space_dimension': 65536, 'coupling_J': 1., 'transverse_field_h': .5},
               'beta_grid': list(BETAS), 'rows': rows[1:], 'infinite_temperature_check': zero,
               'validation_sha256': sha256(output / 'validation.json'),
               'sector_metadata_sha256': {sector_name(s): sha256(checkpoint_paths(output, s)[1])
                                          for s in sectors(4, 4)},
               'spectrum_checks': {'dimension': len(all_values), 'trace_H': float(all_values.sum()),
                                   'trace_H2_per_state': float(all_values@all_values/65536)},
               'uncertainty': 'no sampling/Trotter error; floating-point numerical accuracy only'}
    immutable_json(output / 'exact_4x4_thermal.json', payload)
    print(f'COMPLETE: {output / "exact_4x4_thermal.json"}', flush=True)


def main():
    """Expose help and reject legacy direct solver execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_known_args()  # Retain --help without permitting legacy actions.
    parser.error('Direct solver execution is disabled; use references/run_references.py '
                 '--data-root /external/output [--validate-only] [--threads N]')


if __name__ == '__main__':
    raise SystemExit(main())
