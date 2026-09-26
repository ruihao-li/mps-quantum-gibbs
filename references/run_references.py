#!/usr/bin/env python3
"""Generate exact physical-beta TFIM references without original data.

Full 4x4 ED is explicit through --systems 4x4 (or the default all systems).
--validate-only runs small independent checks, never the 4x4 dense sectors.
Results are immutable and written only under --data-root.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

# Set thread controls before NumPy/SciPy, including in subprocess validators.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--threads', type=int, default=1)
_pre.add_argument('--data-root', type=Path)
_early, _ = _pre.parse_known_args()
if not 1 <= _early.threads <= 12:
    raise ValueError('threads must be between 1 and 12')
for _key in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = str(_early.threads)
if _early.data_root is not None:
    os.environ['MPS_REFERENCE_DATA_ROOT'] = str(_early.data_root.resolve())

import numpy as np
import scipy

try:
    from . import exact_chain, ground_state, exact_thermal
    from .io_utils import immutable_json, read_verified, sha256
except ImportError:  # Direct CLI invocation, independent of working directory.
    import exact_chain
    import ground_state
    import exact_thermal
    from io_utils import immutable_json, read_verified, sha256

HERE = Path(__file__).resolve().parent
CHAIN_BETAS = (.1, .4, 1., 1.1, 1.2, 1.8, 2.1, 3.)
SQUARE_BETAS = (.05, .2, .4, .45, .5, .7, .85, 2.)
SYSTEMS = {'10x1': (10, 1), '20x1': (20, 1), '3x3': (3, 3), '4x4': (4, 4)}
PHYSICS = {'J': 1., 'h': .5, 'boundary': 'open'}


def require(condition, message):
    """Raise AssertionError with the supplied message when a gate fails."""
    if not condition:
        raise AssertionError(message)


def configure(data_root):
    """Configure external reference paths and return their common directory.

    Raises:
        ValueError: The requested data root is inside the code-only release.
    """
    data_root = Path(data_root).expanduser().resolve()
    bundle_root = HERE.parent
    if data_root == bundle_root or bundle_root in data_root.parents:
        raise ValueError('Numerical outputs must be outside the code-only release directory')
    root = data_root / 'references'
    os.environ['MPS_REFERENCE_DATA_ROOT'] = str(data_root)
    exact_thermal.OUTPUT = root / 'exact_4x4'
    exact_thermal.GROUND = root / 'ground_4x4.json'
    return root


def environment():
    """Return runtime versions, platform, and the configured BLAS thread limit."""
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__, 'scipy': scipy.__version__,
            'configured_blas_threads': exact_thermal.THREADS}


def chain_rows(n, betas=CHAIN_BETAS):
    """Build validated exact chain rows at positive physical inverse temperatures.

    Add partition functions, natural-log entropies, and free energies to the
    Majorana-covariance observables, then apply the canonical thermal schema.
    """
    rows = []
    for beta in betas:
        row = exact_chain.exact_observables(n, beta)
        modes = np.asarray(row['single_particle_energies'])
        logz = float(np.sum(np.logaddexp(.5 * beta * modes, -.5 * beta * modes)))
        entropy = beta * row['energy'] + logz
        row.update(log_partition=logz, entropy_nats=entropy, entropy_density_nats=entropy/n,
                   free_energy=-logz/beta, free_energy_density=-logz/(beta*n),
                   dimensionless_free_energy_density=-logz/n,
                   thermodynamic_identity_residual=beta*row['energy']-entropy+logz,
                   mean_longitudinal_magnetization=0.)
        rows.append(canonical_row(row))
    return rows


def dense_square_rows(side, betas=SQUARE_BETAS):
    """Compute complete dense thermal references for a 2x2 or 3x3 square.

    Returns:
        Canonical thermal rows with all Pauli ZiZj correlations and the full
        ordered spectrum, using the same solver as the original 3x3 driver.
    """
    require(side in (2, 3), 'Only inexpensive 2x2/3x3 dense construction is allowed')
    h, z = exact_thermal.full_system(side, side)
    energies, vectors = np.linalg.eigh(h.toarray())
    probabilities = np.square(vectors)
    n = side * side
    diag = np.concatenate(((z.sum(axis=0)**2)[None, :],
                           np.einsum('ia,ja->ija', z, z).reshape(n*n, -1)))
    weights = np.einsum('ij,jk->ik', diag, probabilities, optimize=False)
    rows = exact_thermal.thermal_rows(energies, weights, betas, n)
    return [canonical_row(row) for row in rows], energies


def canonical_row(row):
    """Add release aliases and symmetry metadata, then validate a thermal row.

    Return a shallow copy. The chi alias denotes an equal-time magnetic
    fluctuation, and C_beta is (beta E-S)/N with entropy in nats.
    """
    row = dict(row)
    n = int(row['num_spins'])
    row.update(C_beta=row['dimensionless_free_energy_density'],
               chi=row['susceptibility_per_spin'], cv=row['specific_heat_per_spin'],
               entropy_density=row['entropy_density_nats'],
               correlations=row['longitudinal_correlations'],
               z_expectations=[0.] * n, norm=1.,
               norm_definition='trace-one analytically normalized Gibbs state')
    validate_thermal(row)
    return row


def validate_thermal(row):
    """Check a canonical thermal row's scalar, correlation, and identity gates.

    Require Pauli onsite normalization, correlation positivity, entropy bounds,
    and the spin-inversion-symmetric equal-time fluctuation formulas.

    Raises:
        AssertionError: A numerical or canonical-schema gate fails.
    """
    n, beta = int(row['num_spins']), float(row['beta'])
    values = [row[key] for key in ('energy', 'energy_variance', 'entropy_nats',
               'log_partition', 'C_beta', 'chi', 'cv', 'norm')]
    require(np.isfinite(values).all(), 'Nonfinite thermal scalar')
    corr = np.asarray(row['correlations'], dtype=float)
    require(corr.shape == (n, n) and np.isfinite(corr).all(), 'Malformed thermal correlations')
    require(np.max(np.abs(corr-corr.T)) < 2e-10, 'Correlation exchange symmetry failed')
    require(np.max(np.abs(np.diag(corr)-1)) < 2e-10, 'Pauli onsite correlations failed')
    require(np.max(np.abs(corr)) <= 1+2e-10, 'Correlation magnitude bounds failed')
    require(np.linalg.eigvalsh(corr).min() >= -2e-10, 'Correlation matrix not PSD')
    require(-2e-10 <= row['entropy_nats'] <= n*np.log(2)+2e-10, 'Entropy bounds failed')
    require(row['energy_variance'] >= -2e-10, 'Negative energy variance')
    require(abs(row['C_beta']-(beta*row['energy']-row['entropy_nats'])/n) < 2e-10,
            'C_beta thermodynamic identity failed')
    require(abs(row['C_beta']+row['log_partition']/n) < 2e-10, 'Partition identity failed')
    require(abs(row['chi']-beta*float(corr.sum())/n) < 2e-9, 'Chi/correlation sum failed')
    require(abs(row['cv']-beta**2*row['energy_variance']/n) < 2e-10, 'Specific heat failed')
    require(abs(row['energy_density']-row['energy']/n) < 2e-10, 'Energy density failed')
    require(abs(row['entropy_density_nats']-row['entropy_nats']/n) < 2e-10, 'Entropy density failed')
    require(row['norm'] == 1. and row['z_expectations'] == [0.] * n, 'Symmetry/normalization failed')


def validate_ground(record, solver_checks=None):
    """Apply ground-comparator gates for one chain or square reference.

    Check zero entropy and physical energy variance, an eight-beta pure-state
    envelope, and geometry-specific covariance or independent solver checks.

    Returns:
        A passed-gate summary; invalid records raise AssertionError.
    """
    n = record['num_spins']
    require(math.isfinite(record['energy']) and record['energy'] < 0, 'Invalid ground energy')
    require(abs(record['energy_decomposition_residual']) <= 1e-10, 'Ground energy decomposition')
    corr = np.asarray(record['longitudinal_correlations'])
    require(corr.shape == (n, n) and np.isfinite(corr).all(), 'Ground correlation shape/finite')
    require(np.max(np.abs(corr-corr.T)) <= 1e-12, 'Ground correlation symmetry')
    require(np.max(np.abs(np.diag(corr)-1)) <= 1e-12, 'Ground correlation onsite')
    require(np.max(np.abs(corr)) <= 1+1e-12, 'Ground correlation bounds')
    require(record['entropy_nats'] == 0 and record['energy_variance'] == 0, 'Pure comparator')
    require(len(record['pure_state_envelope']) == 8, 'Ground beta envelope length')
    for row in record['pure_state_envelope']:
        require(abs(row['dimensionless_objective_C_beta']-row['beta']*record['energy']/n) <= 5e-15,
                'Ground objective envelope')
    if record['geometry'] == 'open_square':
        require(abs(record['state_norm']-1) <= 1e-12, 'Ground state norm')
        require(record['eigenpair_residual_l2'] <= 1e-10, 'Ground eigenpair residual')
        require(abs(record['parity_expectation']-1) <= 1e-10, 'Ground even parity')
        require(abs(record['mean_longitudinal_magnetization']) <= 1e-10, 'Ground spin inversion')
        require(solver_checks is not None, 'Missing independent square solver checks')
        for key in ('independent_even_energy_difference_abs', 'full_hilbert_energy_difference_abs',
                    'ritz_vs_rayleigh_difference_abs'):
            require(math.isfinite(solver_checks[key]) and solver_checks[key] <= 1e-10, key)
    else:
        require(record['covariance_antisymmetry_residual_max_abs'] <= 1e-12, 'Ground covariance skew')
        require(record['covariance_purity_residual_max_abs'] <= 1e-11, 'Ground covariance purity')
    return {'passed': True, 'checks': 'original per-system ground-state numerical gates'}


def ground_for(system):
    """Generate and validate one system's exact ground-state comparator.

    Return the ground record, solver checks, and validation report. Both chain
    sizes require the independent N=10 spin-basis/covariance cross-check.
    """
    nx, ny = SYSTEMS[system]
    if ny == 1:
        record = ground_state.zero_temperature_chain(exact_chain, nx, CHAIN_BETAS)
        checks = {}
    else:
        record, checks = ground_state.square_ground_state(nx, SQUARE_BETAS)
    validation = validate_ground(record, checks)
    # Spin-basis N10 cross-check stays mandatory for both chain sizes.
    if ny == 1:
        validation['chain_N10_spin_basis_crosscheck'] = validate_chain_ground()
    return record, checks, validation


def canonical_ground(record):
    """Add release aliases to a ground comparator without making it thermal.

    Return a shallow copy with beta-envelope rows, chi/beta from the fixed
    ground-state magnetic variance, and state-normalization metadata.
    """
    result = dict(record)
    n = result['num_spins']
    result.update(correlations=result['longitudinal_correlations'], z_expectations=[0.] * n,
                  norm=result.get('state_norm', 1.), chi_over_beta=result['magnetization_variance_per_spin'],
                  norm_definition='state norm for sparse state; pure covariance construction for chains')
    result['rows'] = [dict(row, C_beta=row['dimensionless_objective_C_beta'],
        chi=row['magnetization_fluctuation_chi_ET'], cv=row['energy_fluctuation_cV_fluc'])
        for row in result['pure_state_envelope']]
    return result


def validate_chain_ground():
    """Cross-check N=10 ground energy and correlations in two representations."""
    reference = ground_state.zero_temperature_chain(exact_chain, 10, CHAIN_BETAS)
    edges = ground_state.system_edges('chain', 10)
    energy, vector = ground_state.solve_sector(10, edges, parity=1, levels=1)
    spin = ground_state.spin_basis_observables(ground_state.expand_parity_vector(vector[:, 0], 1),
            ground_state.full_hamiltonian_csr(10, edges), edges)
    de = abs(float(energy[0])-reference['energy'])
    dc = float(np.max(np.abs(np.asarray(spin['longitudinal_correlations'])-
                            reference['longitudinal_correlations'])))
    require(de <= 1e-11 and dc <= 1e-10, 'Independent N10 ground spin/covariance cross-check')
    return {'energy_abs_error': de, 'correlation_max_abs_error': dc}


def validate_chain_thermal():
    """Cross-check N=4 chain covariance thermodynamics against dense spin ED."""
    n, beta = 4, .73
    edges = ground_state.system_edges('chain', n)
    matrix = ground_state.full_hamiltonian_csr(n, edges).toarray()
    energies, vectors = np.linalg.eigh(matrix)
    p = np.exp(-beta*(energies-energies[0])); p /= p.sum()
    basis = np.arange(1 << n)
    z = 1.-2.*((basis[None, :] >> np.arange(n)[:, None]) & 1)
    populations = np.square(vectors) @ p
    corr = np.einsum('ia,ja,a->ij', z, z, populations, optimize=False)
    energy = float(p @ energies)
    variance = float(p @ (energies-energy)**2)
    row = chain_rows(n, [beta])[0]
    expected = {'energy': energy, 'energy_variance': variance,
                'chi': beta*float(corr.sum())/n, 'cv': beta**2*variance/n}
    for key, value in expected.items():
        require(math.isclose(row[key], value, abs_tol=2e-12, rel_tol=2e-12),
                'Independent dense N4 chain thermal check: '+key)
    require(np.max(np.abs(np.asarray(row['correlations'])-corr)) < 2e-12,
            'Independent dense N4 chain correlations')
    return {'passed': True, 'N': n, 'beta': beta, 'absolute_tolerance': 2e-12}


def expected_contract(system):
    """Return the current system, physics, source, and runtime reuse contract."""
    nx, ny = SYSTEMS[system]
    return {'schema_version': 1, 'dataset_kind': 'exact_reference_system', 'system': system,
            'geometry': 'open_chain' if ny == 1 else 'open_square', 'nx': nx, 'ny': ny,
            'N': nx*ny, 'physics': PHYSICS, 'beta_grid': list(CHAIN_BETAS if ny == 1 else SQUARE_BETAS),
            'source_sha256': exact_thermal.source_hashes(), 'environment': environment()}


def check_system(payload, system):
    """Check a saved system contract, beta coverage, and thermal/ground gates."""
    for key, value in expected_contract(system).items():
        require(payload.get(key) == value, 'Existing reference contract differs: '+key)
    require(payload.get('passed') is True, 'Reference validation absent')
    require([r['beta'] for r in payload['thermal']] == payload['beta_grid'], 'Thermal beta coverage')
    for row in payload['thermal']:
        validate_thermal(row)
    validate_ground(payload['ground'], payload['ground_solver_checks'])


def generate_small_system(root, system):
    """Reuse or immutably generate a chain or 3x3 reference with cross-checks.

    Existing JSON must pass its digest and current contract checks. New output
    combines exact thermal rows with a separately validated ground comparator.
    """
    path = root / (system + '.json')
    if path.exists():
        payload = read_verified(path)
        check_system(payload, system)
        return payload
    nx, ny = SYSTEMS[system]
    require(system != '4x4', 'Use symmetry-resolved solver for 4x4')
    rows, energies = (chain_rows(nx), None) if ny == 1 else dense_square_rows(nx)
    ground, checks, validation = ground_for(system)
    if energies is not None:
        delta = abs(float(energies[0])-ground['energy'])
        require(delta <= 1e-11, 'Independent 3x3 dense/parity-Lanczos ground check')
        validation['dense_ground_energy_abs_error'] = delta
    else:
        thermal_e0 = -.5 * sum(rows[0]['single_particle_energies'])
        require(abs(thermal_e0-ground['energy']) <= 1e-12, 'Thermal modes/ground energy check')
    payload = {**expected_contract(system), 'passed': True, 'thermal': rows,
               'ground': canonical_ground(ground), 'ground_solver_checks': checks,
               'validation': validation}
    check_system(payload, system)
    immutable_json(path, payload)
    return payload


def prepare_ground_4x4(root):
    """Reuse or publish the source-bound independent 4x4 ground comparator."""
    path = root / 'ground_4x4.json'
    if path.exists():
        exact_thermal.ground_reference()
        return read_verified(path)
    ground, checks, validation = ground_for('4x4')
    payload = {'schema_version': 1, 'dataset_kind': 'exact_ground_reference',
               'system': '4x4', 'physics': PHYSICS, 'passed': True,
               'source_sha256': exact_thermal.source_hashes(), 'environment': environment(),
               'ground': canonical_ground(ground), 'solver_checks': checks, 'validation': validation}
    immutable_json(path, payload)
    exact_thermal.ground_reference()
    return payload


def generate_4x4(root):
    """Reuse or generate a complete symmetry-resolved 4x4 thermal reference.

    New output requires fresh 3x3 and ground references, sector validation,
    and all eight complete spectra. Existing output is independently retraced
    from verified sector checkpoints before reuse.
    """
    path = root / '4x4.json'
    if path.exists():
        payload = read_verified(path)
        check_system(payload, '4x4')
        # Verify every source-bound checkpoint and independently retrace it.
        verify_4x4(payload)
        return payload
    generate_small_system(root, '3x3')  # Current-run independent reference, no original fixture.
    ground = prepare_ground_4x4(root)
    exact_thermal.validate(exact_thermal.OUTPUT)
    exact_thermal.run_all(exact_thermal.OUTPUT)
    raw = read_verified(exact_thermal.OUTPUT / 'exact_4x4_thermal.json')
    payload = {**expected_contract('4x4'), 'passed': True,
               'thermal': [canonical_row(row) for row in raw['rows']],
               'ground': ground['ground'], 'ground_solver_checks': ground['solver_checks'],
               'validation': {'passed': True, 'complete_spectrum_dimension': 65536,
                   'sector_validation_sha256': sha256(exact_thermal.OUTPUT / 'validation.json'),
                   'ground_reference_sha256': sha256(root / 'ground_4x4.json')},
               'infinite_temperature_check': raw['infinite_temperature_check']}
    check_system(payload, '4x4')
    verify_4x4(payload)
    immutable_json(path, payload)
    return payload


def verify_4x4(payload):
    """Retrace saved 4x4 thermal results from all eight verified checkpoints.

    Check full-spectrum trace identities and recompute energies, entropies,
    heat capacities, and correlations without another eigendecomposition.
    """
    exact_thermal.verify_validation(exact_thermal.OUTPUT)
    values, weights = [], []
    for signs in exact_thermal.sectors(4, 4):
        checkpoint = exact_thermal.load_checkpoint(exact_thermal.OUTPUT, signs)
        require(checkpoint is not None, 'Missing exact 4x4 sector')
        e, w, _ = checkpoint
        values.append(e); weights.append(w)
    e, w = np.concatenate(values), np.concatenate(weights, axis=1)
    require(len(e) == 65536 and abs(e.sum()) < 2e-8 and abs(np.mean(e*e)-28) < 2e-10,
            'Full 4x4 spectrum trace identities')
    for row in payload['thermal']:
        beta = row['beta']
        p = np.exp(-beta*(e-e.min())); p /= p.sum()
        energy = float(np.sum(p*e))
        entropy = float(-np.sum(p*np.log(np.maximum(p, np.finfo(float).tiny))))
        corr = np.einsum('ij,j->i', w[1:], p, optimize=False).reshape(16, 16)
        cv = beta**2*float(np.sum(p*(e-energy)**2))/16
        for key, value in [('energy', energy), ('entropy_nats', entropy), ('cv', cv)]:
            require(math.isclose(row[key], value, abs_tol=5e-11, rel_tol=3e-12),
                    'Independent saved-spectrum retrace: '+key)
        require(np.allclose(row['correlations'], corr, atol=5e-11, rtol=3e-12),
                'Independent saved-spectrum correlation retrace')


def validate_small(root):
    """Publish an immutable audit of independent chain and small-square checks.

    Generate or reuse the 3x3 input required by the square validator. No 4x4
    dense eigendecomposition is performed. Return the audit JSON path.
    """
    generate_small_system(root, '3x3')
    chain_test = validate_chain_thermal()
    chain_ground_test = validate_chain_ground()
    command = [sys.executable, str(HERE / 'validate_square.py'), '--threads', str(exact_thermal.THREADS)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    sys.stderr.write(completed.stderr)
    square_test = json.loads(completed.stdout)
    require(square_test['status'] == 'PASS' and square_test['tests_run'] == 5,
            'Independent square validation coverage')
    report = {'passed': True, 'source_sha256': exact_thermal.source_hashes(),
              'environment': environment(), 'chain_thermal': chain_test,
              'chain_ground': chain_ground_test, 'square': square_test,
              'scope': 'Small independent checks only; no 4x4 dense eigendecomposition'}
    # Runtime measurements differ per invocation: retain a separate immutable audit.
    import hashlib
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]
    path = root / 'validation' / ('small_' + digest + '.json')
    immutable_json(path, report)
    return path


def main(argv=None):
    """Run the reference-generation CLI or its small-validation-only mode.

    Args:
        argv: Argument sequence, or None to read the process command line.

    Returns:
        Zero after successful validation or immutable bundle publication.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--systems', nargs='+', choices=tuple(SYSTEMS), default=list(SYSTEMS))
    parser.add_argument('--threads', type=int, default=_early.threads)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.threads != exact_thermal.THREADS:
        parser.error('Changing numerical thread configuration requires a fresh CLI process')
    if len(set(args.systems)) != len(args.systems):
        parser.error('Duplicate system selection')
    try:
        root = configure(args.data_root)
    except ValueError as error:
        parser.error(str(error))
    validate_chain_thermal()  # Required even for production, never only a developer test.
    if args.validate_only:
        print('PASS independent reference validation:', validate_small(root))
        return 0
    systems = {}
    for system in args.systems:
        print('Reference:', system, flush=True)
        systems[system] = generate_4x4(root) if system == '4x4' else generate_small_system(root, system)
    bundle = {'schema_version': 1, 'dataset_kind': 'exact_reference_bundle',
              'physics': PHYSICS, 'source_sha256': exact_thermal.source_hashes(),
              'environment': environment(), 'systems': systems,
              'system_files_sha256': {system+'.json': sha256(root/(system+'.json')) for system in systems},
              'conventions': {'beta': 'physical inverse temperature; no ln(2) conversion',
                 'entropy': 'natural logarithms, nats', 'site_order': 'row-major; site0 least-significant bit',
                 'chi': 'beta Var(Mz)/N, equal-time fluctuation, NOT Kubo susceptibility',
                 'cv': 'beta^2 Var(H)/N', 'C_beta': '(beta E-S)/N'}}
    path = root / 'reference_bundle.json'
    immutable_json(path, bundle)
    print('COMPLETE:', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
