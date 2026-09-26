#!/usr/bin/env python3
"""Fresh, portable final-paper HEA/small campaigns; no saved-data prerequisites."""
from __future__ import annotations
import common  # installs thread policy before any numerical imports
from common import np, read, digest, immutable_json, parameter_hash
from common import environment, sources, output_path, verify_directory, publish_directory
from dataclasses import asdict, replace
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import queue
import time
from protocols import hea_settings, small_settings, make_spec, options, start
from evaluation import score
from gibbs_optimizers import optimize_adam_lbfgsb


def campaign(experiment):
    """Describe the 20-start protocol with source, environment, and selection bindings."""
    settings = hea_settings() if experiment == "hea" else small_settings()
    return dict(schema_version=1, experiment=experiment, starts_per_setting=20,
        settings=settings, options=asdict(options(experiment)), sources=sources(),
        environment=environment(), reproduction="protocol-equivalent-current-kernels",
        selection="minimum independently reconstructed best_metrics.C_beta across 20 starts")


def binding(setting, index, theta, experiment, manifest):
    """Hash a run's campaign, setting, start index, and initial parameter bytes."""
    return digest(dict(campaign=manifest, setting=setting, start_index=index,
        initial_parameter_sha256=parameter_hash(theta), experiment=experiment))


def trace_arrays(result):
    """Convert noninitial optimizer trace points to named telemetry arrays.

    Preserve phase-local and global evaluation indices. C_beta fields use
    the optimizer's intensive objective (beta * E - S) / N, while energy
    and entropy fields remain totals.
    """
    rows = [p for p in result.trace if p.phase != "initial"]
    return dict(phase=np.asarray([p.phase for p in rows], dtype="U8"),
        global_call_index=np.asarray([p.call_index for p in rows]),
        phase_local_index=np.asarray([p.phase_call_index for p in rows]),
        energy=np.asarray([p.metrics.energy for p in rows]),
        entropy_nats=np.asarray([p.metrics.entropy_nats for p in rows]),
        C_beta=np.asarray([p.objective_value for p in rows]),
        best_C_beta=np.asarray([p.best_objective_value for p in rows]),
        gradient_norm_2=np.asarray([p.gradient_norm for p in rows]),
        gradient_norm_inf=np.asarray([p.gradient_norm_inf for p in rows]),
        elapsed_seconds=np.asarray([p.elapsed_seconds for p in rows]),
        phase_elapsed_seconds=np.asarray([p.phase_elapsed_seconds for p in rows]))


def run_one(root, experiment, setting, index, manifest):
    """Optimize one cold start and publish independently reconstructed endpoints.

    Reuse an existing directory only after checking its binding and complete
    file inventory. Otherwise, score both the best-observed and terminal
    optimizer parameters independently, retain their distinct metrics, and
    publish the run record together with its evaluation trace.

    Returns:
        The saved run record. Optimizer timing excludes compilation, the
        initial objective-only call, and independent endpoint reconstruction.
    """
    theta, seed = start(setting, index, experiment)
    bound = binding(setting, index, theta, experiment, manifest)
    target = Path(root)/"runs"/setting["setting_id"]/f"start_{index:03d}"
    if target.exists():
        verify_directory(target, bound)
        return read(target/"record.json")
    result = optimize_adam_lbfgsb(make_spec(setting, experiment), theta, options(experiment))
    clock = time.perf_counter()
    best = score(setting, result.parameters, experiment, result.internal_metrics)
    final = score(setting, result.terminal_parameters, experiment, result.terminal_metrics)
    reconstruction_seconds = time.perf_counter()-clock
    fields = ("adam_evaluations", "adam_chunks", "adam_stop_reason", "adam_watchdog_triggered",
        "adam_handoff_consecutive_checks", "adam_handoff_objective_gain", "lbfgs_evaluations",
        "lbfgs_iterations", "lbfgs_nfev", "lbfgs_njev", "lbfgs_stop_reason", "lbfgs_watchdog_triggered",
        "lbfgs_start_source", "optimizer_status", "optimizer_message", "optimizer_success",
        "terminal_projected_gradient_inf", "effective_lbfgs_gtol", "effective_lbfgs_maxfun",
        "compile_seconds", "compile_cache_hit", "adam_seconds", "lbfgs_seconds", "total_seconds")
    diagnostics = {key: getattr(result, key) for key in fields}
    diagnostics.update(objective_evaluations=result.adam_evaluations+result.lbfgs_evaluations,
        optimizer_seconds=result.adam_seconds+result.lbfgs_seconds,
        reconstruction_seconds=reconstruction_seconds, converged=result.criteria_converged,
        timing_excludes="compilation, initial objective-only call, and independent reconstruction")
    record = dict(schema_version=1, experiment=experiment, setting_id=setting["setting_id"],
        setting=setting, start_index=index, seed=seed, campaign_sha256=manifest,
        initial_parameters=theta.tolist(), best_parameters=result.parameters.tolist(),
        final_parameters=result.terminal_parameters.tolist(),
        initial_parameter_sha256=parameter_hash(theta), best_metrics=best, final_metrics=final,
        diagnostics=diagnostics, trace_file="trace.npz")
    publish_directory(target, bound, record, {"trace.npz": trace_arrays(result)})
    return record


def worker(task):
    """Process one setting's assigned starts and queue progress after each completion."""
    root, experiment, setting, indices, manifest, progress_queue = task
    for index in indices:
        run_one(root, experiment, setting, index, manifest)
        progress_queue.put(1)
    return len(indices)


def validate(experiment):
    """Check generated small states, directional derivatives, and optimizer plumbing.

    Compare JAX thermodynamics with independent reconstruction, differentiate
    the intensive C_beta objective, and run a deliberately tiny optimizer
    smoke check. No production shapes or historical trajectories are replayed.
    """
    import jax
    import jax.numpy as jnp
    backend = __import__("jax_dense_gibbs" if experiment == "small" else "jax_mps_gibbs")
    if experiment == "small":
        settings = [s for s in small_settings() if s["num_system"] == 4 and s["beta_index"] == 3]
    else:
        settings = []
        for family in ("contiguous", "interleaved_pair"):
            s = dict(hea_settings()[0], setting_id="validation_"+family, system="2x2", nx=2, ny=2,
                num_system=4, num_ancillas=2, num_layers=2, ansatz_type=family, beta=.7)
            settings.append(s)
    residuals = []
    rng = np.random.default_rng(71291)
    for setting in settings:
        spec = make_spec(setting, experiment)
        theta = rng.uniform(.2, 2., spec.num_parameters)
        values = backend.thermodynamics(jnp.asarray(theta), spec)
        trusted = score(setting, theta, experiment)
        error = max(abs(float(getattr(values, key))-trusted[key])
                    for key in ("energy", "entropy_nats", "free_energy", "norm"))
        if error > 2e-9:
            raise AssertionError(f"Validation forward mismatch {error}")
        def objective(params):
            """Return C_beta = (beta * E - S) / N for the validation setting."""
            v = backend.thermodynamics(params, spec)
            return (spec.beta*v.energy-v.entropy_nats)/spec.num_system
        direction = rng.normal(size=theta.size)
        direction /= np.linalg.norm(direction)
        analytic = float(jax.grad(objective)(jnp.asarray(theta)) @ direction)
        step = 1e-5
        numeric = float((objective(theta+step*direction)-objective(theta-step*direction))/(2*step))
        if not np.isclose(analytic, numeric, atol=2e-7, rtol=2e-5):
            raise AssertionError("Directional gradient validation failed")
        residuals.append(dict(setting=setting["setting_id"], forward=error,
                              directional_derivative=abs(analytic-numeric)))
    # Tiny optimizer plumbing, separately identified and never a paper run.
    smoke_options = replace(options(experiment), adam_min_steps=2, adam_chunk_steps=1,
        adam_handoff_window=1, adam_handoff_patience=1, adam_handoff_tolerance_per_site=1e3,
        adam_watchdog_steps=3, lbfgs_maxiter=2, lbfgs_gtol_per_site=1e3)
    smoke = optimize_adam_lbfgsb(spec, theta, smoke_options)
    score(setting, smoke.parameters, experiment, smoke.internal_metrics)
    return dict(status="passed", scope="generated small systems; no production shapes or historical replay",
                checks=residuals, optimizer_smoke_converged=smoke.criteria_converged)


def main(argv=None):
    """Plan, validate, or launch a portable optimization campaign from the CLI.

    Require current source/environment-bound validation before launch, and
    preserve the production protocol when limiting execution to a run prefix.
    Multiple workers use spawned processes and report completed-run progress.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("hea", "small"))
    parser.add_argument("--output-root", required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--validate", action="store_true")
    actions.add_argument("--launch", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-runs", type=int, help="Run a bounded prefix, retaining exact production protocol")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit_runs is not None and args.limit_runs < 1):
        parser.error("worker and run limits must be positive")
    root = output_path(args.output_root, args.experiment)
    config = campaign(args.experiment)
    manifest = digest(config)
    if args.plan:
        print(common.encoded(dict(settings=len(config["settings"]),
            runs=20*len(config["settings"]), output=root, workers=args.workers,
            options=config["options"], settings_matrix=config["settings"])).decode())
        return
    if args.validate:
        report = dict(validate(args.experiment), campaign_sha256=manifest)
        immutable_json(root/"validation.json", report)
        print(common.encoded(report).decode())
        return
    validation = read(root/"validation.json")
    if validation.get("status") != "passed" or validation.get("campaign_sha256") != manifest:
        raise RuntimeError("Run --validate with current source/environment before launching")
    immutable_json(root/"campaign.json", config)
    tasks = []
    remaining = args.limit_runs or 20*len(config["settings"])
    for setting in config["settings"]:
        indices = list(range(min(20, remaining)))
        if not indices:
            break
        tasks.append((root, args.experiment, setting, indices, manifest))
        remaining -= len(indices)
    from tqdm.auto import tqdm
    with tqdm(total=sum(len(t[3]) for t in tasks), desc=args.experiment+" runs") as progress:
        if args.workers == 1:
            for root_, exp, setting, indices, bound in tasks:
                for index in indices:
                    run_one(root_, exp, setting, index, bound)
                    progress.update(1)
        else:
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                progress_queue = manager.Queue()
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
                    pending = {pool.submit(worker, (*task, progress_queue)) for task in tasks}
                    while pending:
                        finished, pending = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                        while True:
                            try:
                                progress.update(progress_queue.get_nowait())
                            except queue.Empty:
                                break
                        for future in finished:
                            future.result()


if __name__ == "__main__":
    main()
