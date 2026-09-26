#!/usr/bin/env python3
"""Finite-size gradients: 12 shapes, 128 fresh states each; no optimization."""
from __future__ import annotations
import common
from common import np, read, digest, immutable_json, environment, sources, output_path
from common import verify_directory, publish_directory, parameter_hash
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from protocols import gradient_shapes, make_spec, start


def kernel(spec):
    """Build a JIT callable for paired energy/entropy derivatives and forward diagnostics."""
    import jax
    import jax.numpy as jnp
    from jax_mps_gibbs import thermodynamics
    def energy_entropy_with_aux(parameters):
        """Return energy and entropy with duplicated values and MPS diagnostics as auxiliaries."""
        values = thermodynamics(parameters, spec)
        energy_entropy = jnp.stack((values.energy, values.entropy_nats))
        diagnostics = jnp.stack((values.norm, values.minimum_schmidt_weight))
        return energy_entropy, (energy_entropy, diagnostics)
    return jax.jit(jax.jacrev(energy_entropy_with_aux, has_aux=True))


def evaluate(compiled, theta):
    """Evaluate a paired-gradient kernel, synchronize execution, and check its outputs.

    Return energies, entropies, both parameter gradients, MPS diagnostics,
    and synchronized call time. Compilation is included if the caller has
    not already warmed the kernel.
    """
    import jax
    import jax.numpy as jnp
    begin = time.perf_counter()
    result = compiled(jnp.asarray(theta, dtype=jnp.float64))
    result = jax.tree_util.tree_map(lambda a: a.block_until_ready(), result)
    jacobian, (values, diagnostics) = result
    jacobian, values, diagnostics = map(np.asarray, (jacobian, values, diagnostics))
    if jacobian.shape != (2, len(theta)) or not all(np.isfinite(v).all() for v in (jacobian, values, diagnostics)):
        raise AssertionError("Invalid gradient kernel output")
    if abs(float(diagnostics[0])-1) > 2e-9:
        raise AssertionError("Gradient MPS norm failed")
    return dict(energy=values[0], entropy_nats=values[1], norm=diagnostics[0],
        minimum_schmidt_weight=diagnostics[1], grad_energy=jacobian[0],
        grad_entropy_nats=jacobian[1], kernel_wall_seconds=time.perf_counter()-begin)


def run_shape(task):
    """Evaluate or resume a deterministic prefix of one gradient architecture.

    Args:
        task: Tuple of output root, shape metadata, sample count, and campaign
            digest, suitable for a spawned worker process.

    Returns:
        The requested sample count after completion or verified cache reuse.

    Notes:
        Save immutable per-sample outputs and aggregate a complete shape
        only at 128 samples. Warmup is excluded from per-sample timing; no
        optimization or beta-specific resampling is performed.
    """
    root, shape, count, manifest = task
    root = Path(root)
    shape_root = root/"shapes"/shape["shape_id"]
    shape_binding = digest(dict(campaign=manifest, shape=shape, samples=128))
    if (shape_root/"complete.json").exists():
        verify_directory(shape_root, shape_binding)
        return count
    compiled = kernel(make_spec(shape))
    theta, _ = start(shape, 0, "gradient")
    # Compilation and warmup are excluded from per-sample timing.
    begin = time.perf_counter()
    evaluate(compiled, theta)
    compile_seconds = time.perf_counter()-begin
    rows = []
    from tqdm.auto import tqdm
    for index in tqdm(range(count), desc=shape["shape_id"], leave=False):
        theta, seed = start(shape, index, "gradient")
        target = root/"samples"/shape["shape_id"]/f"sample_{index:03d}"
        bound = digest(dict(campaign=manifest, shape=shape, sample=index,
                            parameter_sha256=parameter_hash(theta)))
        if target.exists():
            verify_directory(target, bound)
            with np.load(target/"data.npz", allow_pickle=False) as saved:
                row = {key: saved[key] for key in saved.files}
        else:
            row = dict(evaluate(compiled, theta), theta=theta, sample_index=index,
                theta_sha256=parameter_hash(theta), seed_state_uint32=seed["state_uint32"],
                rng_spawn_key=seed["spawn_key"], rng_master_seed=seed["entropy"])
            publish_directory(target, bound,
                dict(schema_version=1, experiment="gradient", shape=shape, sample_index=index,
                     seed=seed, campaign_sha256=manifest, compile_seconds=compile_seconds),
                {"data.npz": row})
        rows.append(row)
    if count == 128:
        arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
        publish_directory(shape_root, shape_binding,
            dict(schema_version=1, experiment="gradient", shape=shape, samples=128,
                 campaign_sha256=manifest, analysis_betas=[.4, 1., 3.],
                 compile_seconds_this_session=compile_seconds,
                 objective="grad_C_beta=(beta*grad_energy-grad_entropy_nats)/num_system"),
            {"data.npz": arrays}, record_name="metadata.json")
    return count


def configuration():
    """Bind the gradient protocol to current source hashes and numerical environment."""
    return dict(schema_version=1, experiment="gradient", shapes=gradient_shapes(),
        samples_per_shape=128, analysis_betas=[.4, 1., 3.],
        sources=sources(), environment=environment(), master_seed=2026091001)


def validate():
    """Check tiny independent MPS metrics and energy/entropy directional derivatives.

    Exercise both HEA layouts on four system spins and report residuals;
    this does not execute or certify the full production architectures.
    """
    from evaluation import trusted_mps
    rows = []
    for family in ("contiguous", "interleaved_pair"):
        shape = dict(gradient_shapes()[0], setting_id="validation_"+family,
            num_system=4, nx=4, ny=1, num_ancillas=2, num_layers=2, ansatz_type=family)
        theta = np.random.default_rng(761).uniform(.3, 1.9, 12)
        compiled = kernel(make_spec(shape))
        values = evaluate(compiled, theta)
        trusted = trusted_mps(shape, theta)
        for key in ("energy", "entropy_nats"):
            if abs(values[key]-trusted[key]) > 2e-9:
                raise AssertionError("Independent gradient metrics disagree")
        direction = np.random.default_rng(91).normal(size=12)
        direction /= np.linalg.norm(direction)
        plus, minus = evaluate(compiled, theta+1e-5*direction), evaluate(compiled, theta-1e-5*direction)
        residuals = {}
        for value, gradient in (("energy", "grad_energy"), ("entropy_nats", "grad_entropy_nats")):
            analytic = float(values[gradient]@direction)
            numerical = float((plus[value]-minus[value])/2e-5)
            if not np.isclose(analytic, numerical, atol=2e-7, rtol=2e-5):
                raise AssertionError("Gradient directional derivative failed")
            residuals[value] = abs(analytic-numerical)
        rows.append(dict(ansatz_type=family, residuals=residuals))
    return dict(status="passed", checks=rows,
                scope="tiny independent states and gradients; full production shapes not executed")


def main(argv=None):
    """Plan, validate, or launch the source-bound gradient campaign from the CLI.

    Launch requires a passing validation for the current configuration and
    environment. Optional limits select a prefix of the 1536 fresh-state
    samples while preserving their original deterministic seed mapping.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--launch", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, help="Bounded prefix of the 1536 production samples")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit_samples is not None and args.limit_samples < 1):
        parser.error("limits must be positive")
    root = output_path(args.output_root, "gradient")
    config = configuration()
    manifest = digest(config)
    if args.plan:
        print(common.encoded(config).decode())
        return
    if args.validate:
        report = dict(validate(), campaign_sha256=manifest)
        immutable_json(root/"validation.json", report)
        print(common.encoded(report).decode())
        return
    report = read(root/"validation.json")
    if report.get("status") != "passed" or report.get("campaign_sha256") != manifest:
        raise RuntimeError("Run --validate with current code/environment first")
    immutable_json(root/"campaign.json", config)
    tasks = []
    remaining = min(args.limit_samples or 1536, 1536)
    for shape in config["shapes"]:
        count = min(remaining, 128)
        if count:
            tasks.append((root, shape, count, manifest))
        remaining -= count
    from tqdm.auto import tqdm
    with tqdm(total=sum(t[2] for t in tasks), desc="gradient samples") as progress:
        if args.workers == 1:
            for task in tasks:
                progress.update(run_shape(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers,
                    mp_context=multiprocessing.get_context("spawn")) as pool:
                for result in as_completed([pool.submit(run_shape, t) for t in tasks]):
                    progress.update(result.result())


if __name__ == "__main__":
    main()
