#!/usr/bin/env python3
"""Build checked publication CSVs from the portable campaign output schema.

No historical manuscript tables are needed. Each stage requires its complete
dataset; use a different table directory after intentionally changing inputs.
HEA observables are reconstructed only for the 224 best-of-20 endpoints.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
sys.path.insert(0, str(ROOT))
import common  # thread policy before numerical libraries
from common import np, read, sha, digest, immutable_json, verify_directory
import argparse
from collections import Counter
import csv
import io
import os
import tempfile
from analysis.observables import measure, distance_average
from analysis.gradient_statistics import summarize

GLOBAL_METRICS = (r'$C_\beta-C_\beta^{\rm ref}$', r'$E/N$', r'$S/N$')
EXACT = 'Exact reference'
GROUND = 'Exact pure ground state'


def csv_output(root, name, rows):
    """Publish deterministic CSV bytes, accepting only identical existing tables.

    Columns follow first occurrence across rows. Publication uses a flushed
    temporary file and a no-clobber hard link; differing existing bytes raise
    RuntimeError, and empty row collections raise ValueError.
    """
    if not rows:
        raise ValueError(f'No rows for {name}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode()
    path = root/name
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing changed table {path}; choose a fresh --tables-root')
        return
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as out:
        temporary = Path(out.name)
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def load_runs(data_root, experiment):
    """Load a complete manuscript campaign after protocol and integrity checks.

    Args:
        data_root: Parent directory containing the campaign outputs.
        experiment: ``hea`` or ``small`` campaign selector.

    Returns:
        The campaign digest and ordered (setting, run-group) pairs, with
        exactly 20 starts per group. Records gain analysis-only run IDs and
        paths; saved records are not modified.

    Notes:
        Check validation binding, exact protocol settings, deterministic
        starts, completed-file inventories and hashes, and normalized
        thermodynamic fields before exposing runs to table selection.
    """
    from protocols import start, hea_settings, small_settings, options
    from dataclasses import asdict
    root = data_root/experiment
    manifest = read(root/'campaign.json')
    manifest_hash = digest(manifest)
    validation = read(root/'validation.json')
    if validation.get('status') != 'passed' or validation.get('campaign_sha256') != manifest_hash:
        raise ValueError('Missing or mismatched campaign validation')
    expected_settings = 224 if experiment == 'hea' else 80
    expected_matrix=hea_settings() if experiment=='hea' else small_settings()
    if manifest.get('settings')!=expected_matrix or manifest.get('options')!=asdict(options(experiment)):
        raise ValueError('Campaign differs from the manuscript numerical protocol')
    if len(manifest['settings']) != expected_settings or manifest['starts_per_setting'] != 20:
        raise ValueError('Incomplete manuscript campaign specification')
    records = []
    settings = manifest['settings']
    if len({s['setting_id'] for s in settings}) != expected_settings:
        raise ValueError('Duplicate setting IDs')
    expected_dirs = {s['setting_id'] for s in settings}
    if {p.name for p in (root/'runs').iterdir() if not p.name.startswith('.')} != expected_dirs:
        raise ValueError('Incomplete or unexpected setting directories')
    for setting in settings:
        directory = root/'runs'/setting['setting_id']
        if {p.name for p in directory.iterdir() if not p.name.startswith('.')} != {f'start_{i:03d}' for i in range(20)}:
            raise ValueError(f'Expected exactly 20 starts: {directory}')
        group = []
        for index in range(20):
            path = directory/f'start_{index:03d}'
            row = read(path/'record.json')
            expected_theta, expected_seed = start(setting,index,experiment)
            np.testing.assert_array_equal(row['initial_parameters'],expected_theta)
            if row['seed'] != expected_seed:
                raise ValueError(f'Cold-start seed mismatch: {path}')
            bound = digest(dict(campaign=manifest_hash, setting=setting, start_index=index,
                initial_parameter_sha256=common.parameter_hash(row['initial_parameters']),
                experiment=experiment))
            verify_directory(path, bound)
            if (row['setting'] != setting or row['start_index'] != index or
                    row['campaign_sha256'] != manifest_hash or row['experiment'] != experiment):
                raise ValueError(f'Run metadata mismatch: {path}')
            for metrics in (row['best_metrics'], row['final_metrics']):
                n, beta = setting['num_system'], setting['beta']
                values = [metrics[k] for k in ('energy','entropy_nats','C_beta','norm')]
                if not np.isfinite(values).all() or abs(metrics['norm']-1) > 2e-8:
                    raise ValueError('Invalid saved thermodynamics')
                np.testing.assert_allclose(metrics['C_beta'],
                    (beta*metrics['energy']-metrics['entropy_nats'])/n, atol=2e-10, rtol=1e-10)
            row['_run_id'] = setting['setting_id']+f'__start_{index:03d}'
            row['_path'] = path
            group.append(row)
        records.append((setting, group))
    return manifest_hash, records


def candidate(setting):
    """Format an HEA layout label with its ancilla and layer counts."""
    family = 'Contiguous' if setting['ansatz_type']=='contiguous' else 'Interleaved'
    return f"{family} ({setting['num_ancillas']},{setting['num_layers']})"


def selected(group):
    """Select minimum trusted best-endpoint C_beta, breaking ties by start index."""
    return min(group, key=lambda r: (r['best_metrics']['C_beta'], r['start_index']))


def quantiles(values):
    """Return the quartiles, median, and extrema of a nonempty collection."""
    return dict(zip(('q25','median','q75','min','max'),
                    [*np.quantile(values,[.25,.5,.75]),min(values),max(values)]))


def reference_map(data_root):
    """Verify and return all four exact-reference systems and the bundle hash.

    Require checksum-bound system files, the manuscript physics and beta
    grids, valid thermal and ground-state checks, and consistent stored
    longitudinal moments. These are exact-reference records, not observables
    reconstructed from variational endpoints.
    """
    from references.io_utils import read_verified
    from references.run_references import validate_thermal, validate_ground, CHAIN_BETAS, SQUARE_BETAS
    path = data_root/'references/reference_bundle.json'
    bundle = read_verified(path)
    if set(bundle['systems']) != {'10x1','20x1','3x3','4x4'}:
        raise ValueError('All four exact reference systems are required')
    if (bundle.get('schema_version')!=1 or bundle.get('dataset_kind')!='exact_reference_bundle' or
            bundle.get('physics')!={'J':1.,'h':.5,'boundary':'open'}):
        raise ValueError('Wrong reference schema/physics')
    for system,payload in bundle['systems'].items():
        nx,ny=map(int,system.split('x'))
        grid=list(CHAIN_BETAS if ny==1 else SQUARE_BETAS)
        filename=system+'.json'
        source=data_root/'references'/filename
        if bundle['system_files_sha256'][filename]!=sha(source) or read_verified(source)!=payload:
            raise ValueError('Reference system-file binding mismatch')
        if (payload.get('N')!=nx*ny or payload.get('nx')!=nx or payload.get('ny')!=ny or
                payload.get('passed') is not True or payload.get('beta_grid')!=grid or
                payload.get('physics')!=bundle['physics'] or
                [r['beta'] for r in payload['thermal']]!=grid):
            raise ValueError('Reference system contract mismatch')
        for row in payload['thermal']:
            if row['num_spins']!=nx*ny:
                raise ValueError('Reference normalization/system-size mismatch')
            validate_thermal(row)
            np.testing.assert_allclose(row['correlations'],row['longitudinal_correlations'],atol=0,rtol=0)
        validate_ground(payload['ground'],payload['ground_solver_checks'])
        g=payload['ground']
        np.testing.assert_allclose(g['correlations'],g['longitudinal_correlations'],atol=0,rtol=0)
        np.testing.assert_allclose(g['chi_over_beta'],np.sum(g['correlations'])/(nx*ny),atol=2e-10,rtol=2e-10)
    return bundle['systems'], sha(path)


def reference_at(system, beta):
    """Return the unique thermal reference matching beta within absolute tolerance."""
    rows = [r for r in system['thermal'] if abs(r['beta']-beta)<1e-13]
    if len(rows) != 1:
        raise ValueError(f'Missing or duplicate reference beta={beta}')
    return rows[0]


def reference_fields(row):
    """Map exact thermodynamics to table fields with zero sampling-error placeholders.

    The zero error fields denote exact-reference, non-sampling output; they
    do not estimate floating-point or eigensolver error.
    """
    return dict(exact_C_beta=row['C_beta'], exact_energy_density=row['energy_density'],
        exact_entropy_density_nats=row['entropy_nats']/row['num_spins'],
        exact_susceptibility=row['chi'], exact_specific_heat=row['cv'],
        exact_C_beta_se=0., exact_C_beta_halfwidth95=0., exact_energy_density_se=0.,
        exact_entropy_density_nats_se=0., exact_susceptibility_se=0., exact_specific_heat_se=0.,
        exact_specific_heat_halfwidth95=0., reference_kind='exact')


def correlation_rows(system, beta, series, matrix):
    """Build distance-shell rows at the selected chain and square plotting points.

    Average the supplied matrix without subtracting site means. Prepared
    endpoints pass connected covariance; exact-reference curves pass the
    reference bundle's symmetry-equivalent longitudinal moments.
    """
    targets = {'20x1':(.4,1.1,2.1,3.), '4x4':(.2,.45,.85,2.)}
    if system not in targets or beta not in targets[system]:
        return []
    nx, ny = map(int,system.split('x'))
    return [dict(system=system,beta=beta,series=series,
                 direction='horizontal' if ny==1 else 'radial',**row)
            for row in distance_average(matrix,nx,ny)]


def hea_tables(data_root, tables):
    """Export checked HEA endpoint, reference, robustness, and resource tables.

    Reconstruct observables only for the minimum trusted C_beta endpoint
    among each setting's 20 starts. Reuse endpoint caches only when their
    campaign, run-record, source bindings, and self-checksum agree. Preserve
    all-start diagnostics separately from the selected curves.

    Args:
        data_root: Parent of complete HEA and exact-reference datasets.
        tables: Destination directory for immutable CSV tables.

    Returns:
        Input campaign and exact-reference bundle hashes for provenance.
    """
    from evaluation import reconstruct_state
    from tqdm.auto import tqdm
    binding, groups = load_runs(data_root,'hea')
    refs, reference_hash = reference_map(data_root)
    output = {name:[] for name in ('shared_global','shared_observables','shared_correlations',
        'shared_correlations_4x4_radial','shared_winners','shared_references','ground_state_comparators',
        'entropy_capacity_vs_target','all_start_diagnostics','optimizer_robustness','mps_resources')}
    bonds, timing = {}, {}
    def append_curves(system,beta,label,energy,entropy,cost,chi,cv,reference_cost,matrix):
        """Append intensive curves and selected-distance correlations to table buffers."""
        n = int(np.prod([int(v) for v in system.split('x')]))
        for metric,value in zip(GLOBAL_METRICS,(cost-reference_cost,energy/n,entropy/n)):
            output['shared_global'].append(dict(system=system,series=label,metric=metric,
                beta=beta,value=value,standard_error=0.))
        for observable,value in (('susceptibility',chi),('specific_heat',cv)):
            output['shared_observables'].append(dict(system=system,series=label,observable=observable,
                beta=beta,value=value,standard_error=0.))
        key = 'shared_correlations' if system.endswith('x1') else 'shared_correlations_4x4_radial'
        output[key].extend(correlation_rows(system,beta,label,matrix))
    for system, reference in refs.items():
        n = int(np.prod([int(v) for v in system.split('x')]))
        ground = reference['ground']
        for row in reference['thermal']:
            beta = row['beta']
            fields = reference_fields(row)
            output['shared_references'].append(dict(system=system,beta=beta,**fields,
                reference_file='references/reference_bundle.json',
                specific_heat_reference_file='references/reference_bundle.json'))
            append_curves(system,beta,EXACT,row['energy'],row['entropy_nats'],row['C_beta'],
                          row['chi'],row['cv'],row['C_beta'],row['correlations'])
            gcost = beta*ground['energy']/n
            gchi = beta*ground['chi_over_beta']
            append_curves(system,beta,GROUND,ground['energy'],0.,gcost,gchi,0.,row['C_beta'],ground['correlations'])
            output['ground_state_comparators'].append(dict(system=system,beta=beta,C_beta=gcost,
                energy_density=ground['energy']/n,entropy_density_nats=0.,susceptibility=gchi,specific_heat=0.))
            output['entropy_capacity_vs_target'].append(dict(system=system,series='Exact target',
                beta=beta,entropy_density_nats=row['entropy_nats']/n))
    for setting, group in tqdm(groups,desc='Best-of-20 endpoint observables'):
        system,beta,n = setting['system'],setting['beta'],setting['num_system']
        label = candidate(setting)
        family = 'contiguous' if setting['ansatz_type']=='contiguous' else 'interleaved'
        na,layers = setting['num_ancillas'],setting['num_layers']
        ref = reference_at(refs[system],beta)
        winner = selected(group)
        metrics = winner['best_metrics']
        cache = data_root/'analysis/endpoints'/(setting['setting_id']+'.json')
        cache_binding = dict(campaign=binding,run_record=sha(winner['_path']/'record.json'),
            observables_source=sha(Path(__file__).with_name('observables.py')),
            evaluation_source=sha(ROOT/'experiments/evaluation.py'))
        if cache.exists():
            measured = read(cache)
            if measured['binding'] != cache_binding:
                raise ValueError(f'Stale endpoint cache: {cache}')
            copy = dict(measured)
            checksum = copy.pop('record_sha256')
            if digest(copy)!=checksum:
                raise ValueError(f'Corrupt endpoint cache: {cache}')
        else:
            state, positions = reconstruct_state(setting,np.asarray(winner['best_parameters']))
            measured = dict(measure(state,positions,setting),binding=cache_binding)
            measured['record_sha256'] = digest(measured)
            immutable_json(cache,measured)
            del state
        np.testing.assert_allclose(measured['energy'],metrics['energy'],atol=2e-8,rtol=2e-10)
        if metrics['C_beta'] < ref['C_beta']-2e-9:
            raise ValueError('Variational objective below exact Gibbs objective')
        append_curves(system,beta,label,metrics['energy'],metrics['entropy_nats'],metrics['C_beta'],
            measured['chi'],measured['cv'],ref['C_beta'],measured['connected'])
        cap = min(na,layers) if family=='contiguous' else na
        if metrics['entropy_nats'] > cap*np.log(2)+2e-8:
            raise ValueError('Entropy exceeds architectural capacity')
        output['entropy_capacity_vs_target'].append(dict(system=system,series=label+' ceiling',
            beta=beta,entropy_density_nats=cap*np.log(2)/n))
        output['shared_winners'].append(dict(setting_id=setting['setting_id'],run_id=winner['_run_id'],
            system=system,family=family,Na=na,L=layers,beta=beta,start_index=winner['start_index'],
            candidate=label,C_beta=metrics['C_beta'],energy_density=metrics['energy']/n,
            entropy_density_nats=metrics['entropy_nats']/n,susceptibility=measured['chi'],
            specific_heat=measured['cv'],achieved_max_bond=measured['max_bond'],**reference_fields(ref)))
        bonds.setdefault((system,label),[]).append(measured['max_bond'])
        runrows = []
        for record in group:
            m,d = record['best_metrics'],record['diagnostics']
            criteria = d['adam_stop_reason']=='handoff_tolerance' and d['lbfgs_stop_reason'] in ('ftol','gtol')
            if bool(d['converged']) != criteria:
                raise ValueError('Inconsistent convergence flag')
            r = dict(run_id=record['_run_id'],setting_id=setting['setting_id'],system=system,
                family=family,candidate=label,Na=na,L=layers,beta=beta,start_index=record['start_index'],
                C_beta=m['C_beta'],energy_density=m['energy']/n,
                total_evaluations=d['objective_evaluations'],optimizer_seconds=d['optimizer_seconds'],
                criteria_converged=criteria,optimizer_success=bool(d['optimizer_success']),
                adam_stop_reason=d['adam_stop_reason'],lbfgs_stop_reason=d['lbfgs_stop_reason'],
                cumulative_discarded_weight=0.,exact_C_beta=ref['C_beta'],
                exact_energy_density=ref['energy_density'],C_gap=m['C_beta']-ref['C_beta'],
                energy_density_error=m['energy']/n-ref['energy_density'])
            runrows.append(r)
            timing.setdefault((system,label),[]).append((d['optimizer_seconds'],d['reconstruction_seconds']))
        output['all_start_diagnostics'].extend(runrows)
        selected_row = runrows[winner['start_index']]
        summary = dict(setting_id=setting['setting_id'],system=system,candidate=label,family=family,
            beta=beta,runs=20,selected_run_id=winner['_run_id'],
            selected_C_gap=selected_row['C_gap'],selected_energy_density_error=selected_row['energy_density_error'],
            criteria_converged_fraction=float(np.mean([r['criteria_converged'] for r in runrows])),
            optimizer_success_fraction=float(np.mean([r['optimizer_success'] for r in runrows])),
            criteria_not_converged_count=sum(not r['criteria_converged'] for r in runrows),
            optimizer_failed_count=sum(not r['optimizer_success'] for r in runrows),
            adam_stop_counts=common.encoded(dict(Counter(r['adam_stop_reason'] for r in runrows))).decode().strip(),
            lbfgs_stop_counts=common.encoded(dict(Counter(r['lbfgs_stop_reason'] for r in runrows))).decode().strip())
        for key in ('C_gap','energy_density_error','total_evaluations','optimizer_seconds'):
            summary.update({key+'_'+q:float(v) for q,v in quantiles([r[key] for r in runrows]).items()})
        output['optimizer_robustness'].append(summary)
    for (system,label), values in bonds.items():
        times = np.asarray(timing[system,label])
        row = dict(system=system,candidate=label,later_execution_cohort=False,
            optimizer_timing_definition='Adam and L-BFGS-B wall time; excludes compilation and reconstruction',
            trusted_timing_definition='independent best and final endpoint reconstruction wall time')
        for name, vals, percentiles in (('bond',values,[0,.5,1]),
                ('optimizer_seconds',times[:,0],[.25,.5,.75]),('trusted_seconds',times[:,1],[.25,.5,.75])):
            row.update({name+'_'+key:float(v) for key,v in zip(('low','median','high'),np.quantile(vals,percentiles))})
        output['mps_resources'].append(row)
    for name,rows in output.items():
        csv_output(tables,name+'.csv',rows)
    return dict(campaign=binding,reference_bundle=reference_hash)


def small_tables(data_root,tables):
    """Export best-of-20 small-system comparisons and return campaign provenance.

    Select by independently reconstructed C_beta, not fidelity, and retain
    the selected endpoint's exact-reference fields and absolute energy error
    per system spin.
    """
    binding, groups = load_runs(data_root,'small')
    rows = []
    for setting, group in groups:
        chosen = selected(group)
        m = chosen['best_metrics']
        row = {k:setting[k] for k in ('setting_id','model','beta','beta_index','num_system','num_ancillas','num_layers')}
        row.update(ansatz=setting['ansatz_type'],selected_replicate=chosen['start_index'],
            energy=m['energy'],entropy_nats=m['entropy_nats'],C_beta=m['C_beta'],
            infidelity=m['infidelity'],fidelity=1-m['infidelity'],exact_energy=m['exact_energy'],
            exact_entropy_nats=m['exact_entropy_nats'],exact_C_beta=m['exact_C_beta'],
            abs_energy_density_error=abs(m['energy']-m['exact_energy'])/setting['num_system'])
        rows.append(row)
    csv_output(tables,'small_tfda_hea.csv',rows)
    return dict(campaign=binding)


def gradient_tables(data_root,tables):
    """Verify all gradient datasets and export paired-bootstrap scaling statistics.

    Require 12 manuscript architectures and 128 deterministic states per
    architecture, checking campaign binding and completed-file hashes before
    summarizing. Return campaign and metadata hashes plus the bootstrap count.
    """
    from protocols import gradient_shapes
    root = data_root/'gradient'
    manifest = read(root/'campaign.json')
    manifest_hash=digest(manifest)
    validation=read(root/'validation.json')
    if validation.get('status')!='passed' or validation.get('campaign_sha256')!=manifest_hash:
        raise ValueError('Missing or mismatched gradient validation')
    expected={s['shape_id']:s for s in manifest['shapes']}
    if manifest['shapes']!=gradient_shapes():
        raise ValueError('Gradient architecture matrix differs from the manuscript protocol')
    if len(expected)!=12 or manifest['samples_per_shape']!=128:
        raise ValueError('Wrong gradient campaign shape count')
    rows, hashes = [], {}
    paths = sorted((root/'shapes').glob('*/metadata.json'))
    if len(paths)!=12:
        raise ValueError('Expected 12 complete gradient shape datasets')
    for path in paths:
        metadata = read(path)
        hashes[str(path.relative_to(data_root))] = sha(path)
        shape = metadata.get('shape')
        if shape is None:
            raise ValueError('Missing gradient shape metadata')
        if (shape!=expected.get(path.parent.name) or metadata.get('samples')!=128 or
                metadata.get('campaign_sha256')!=manifest_hash):
            raise ValueError('Gradient shape/campaign binding mismatch')
        verify_directory(path.parent,digest(dict(campaign=manifest_hash,shape=shape,samples=128)))
        with np.load(path.parent/'data.npz',allow_pickle=False) as arrays:
            if arrays['theta'].shape[0]!=128:
                raise ValueError('Expected 128 gradient samples')
            from protocols import start
            for i,theta in enumerate(arrays['theta']):
                np.testing.assert_array_equal(theta,start(shape,i,'gradient')[0])
            rows.extend(summarize(shape,arrays))
    csv_output(tables,'gradient_scaling.csv',rows)
    return dict(campaign=digest(manifest),shape_metadata=hashes,bootstrap_resamples=5000)


def main(argv=None):
    """Export requested analysis stages outside the release and bind their provenance."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--tables-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('hea','small','gradient','all'),default='all')
    args=parser.parse_args(argv)
    data_root,tables=args.data_root.resolve(),args.tables_root.resolve()
    if ROOT==tables or ROOT in tables.parents:
        parser.error('Tables must be outside the code-only bundle')
    for stage in ('hea','small','gradient') if args.stage=='all' else (args.stage,):
        provenance=globals()[stage+'_tables'](data_root,tables)
        immutable_json(tables/(stage+'_table_manifest.json'),dict(schema_version=1,inputs=provenance,
            sources={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
        print(f'{stage}: tables saved in {tables}',flush=True)


if __name__=='__main__':
    main()
