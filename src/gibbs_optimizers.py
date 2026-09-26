"""Criterion-driven float64 Adam--L-BFGS-B for Gibbs-state objectives.

The experiment launcher supplies each start and handles process scheduling.
Compiled numerical kernels are cached locally for repeated optimizations.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib
from numbers import Integral, Real
import time
from typing import Any, Sequence

import numpy as np


class OptimizationConvergenceError(RuntimeError):
    """Raised when a criterion-driven phase has no valid converged endpoint.

    The default policy raises on a finite work limit; protocol options can
    retain a finite, unconverged endpoint at a limit. Abnormal L-BFGS-B exits
    always raise. Structured attributes identify the phase and stop reason.
    """

    def __init__(
        self,
        message: str,
        phase: str,
        stop_reason: str,
        telemetry: dict[str, Any] | None = None,
    ) -> None:
        """Record the failed phase and copy optional diagnostic telemetry."""
        super().__init__(message)
        self.message = message
        self.phase = phase
        self.stop_reason = stop_reason
        self.telemetry = {} if telemetry is None else dict(telemetry)

    def __reduce__(self):
        """Return constructor arguments for lossless exception pickling."""
        return (
            type(self),
            (self.message, self.phase, self.stop_reason, self.telemetry),
        )


@dataclass(frozen=True)
class AdamLBFGSBOptions:
    """Configuration for the bounded float64 hybrid optimizer.

    With the defaults, Adam minimizes ``C = (beta * E - S) / N`` until the
    50-step running-best gain is at most ``1e-4`` at two consecutive 25-step
    checks.  L-BFGS-B then starts from Adam's best evaluated point and may
    return only after SciPy reports native ``ftol`` or ``gtol`` convergence.
    ``adam_watchdog_steps`` and ``lbfgs_maxiter`` are failure watchdogs, not
    accepted stopping rules.

    The experiment protocols select whether a watchdog raises or retains the
    best available endpoint. A watchdog exit is never counted as convergence.
    """

    adam_watchdog_steps: int = 5000
    adam_learning_rate: float = 0.02
    lower_bound: float = -2.0 * np.pi
    upper_bound: float = 2.0 * np.pi
    lbfgs_maxiter: int = 1500
    lbfgs_maxls: int = 20
    lbfgs_ftol: float = 2.220446049250313e-9
    adam_min_steps: int = 200
    adam_chunk_steps: int = 25
    adam_handoff_window: int = 50
    adam_handoff_tolerance_per_site: float = 1.0e-4
    adam_handoff_patience: int = 2
    lbfgs_gtol_per_site: float = 5.0e-7
    adam_watchdog_action: str = "raise"
    lbfgs_nonconvergence_action: str = "raise"

    def __post_init__(self) -> None:
        """Validate stopping policies, finite bounds, and optimizer limits."""
        if (
            not isinstance(self.adam_watchdog_action, str)
            or self.adam_watchdog_action not in {"raise", "handoff"}
        ):
            raise ValueError(
                "adam_watchdog_action must be 'raise' or 'handoff'"
            )
        if (
            not isinstance(self.lbfgs_nonconvergence_action, str)
            or self.lbfgs_nonconvergence_action
            not in {"raise", "retain_limit"}
        ):
            raise ValueError(
                "lbfgs_nonconvergence_action must be 'raise' or "
                "'retain_limit'"
            )
        if isinstance(self.adam_learning_rate, bool) or not isinstance(
            self.adam_learning_rate, Real
        ):
            raise TypeError("adam_learning_rate must be a real scalar")
        if not np.isfinite(self.adam_learning_rate) or self.adam_learning_rate <= 0:
            raise ValueError("adam_learning_rate must be finite and positive")
        if not np.isfinite(self.lower_bound) or not np.isfinite(self.upper_bound):
            raise ValueError("parameter bounds must be finite")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (
                self.lbfgs_maxiter,
                self.lbfgs_maxls,
                self.adam_watchdog_steps,
                self.adam_min_steps,
                self.adam_chunk_steps,
                self.adam_handoff_window,
                self.adam_handoff_patience,
            )
        ):
            raise TypeError("optimizer limits and Adam windows must be integers")
        if self.lbfgs_maxiter <= 0 or self.lbfgs_maxls <= 0:
            raise ValueError("invalid L-BFGS-B iteration/line-search limit")
        if self.adam_watchdog_steps <= 0:
            raise ValueError("adam_watchdog_steps must be positive")
        if self.adam_min_steps < 0:
            raise ValueError("adam_min_steps must be nonnegative")
        if (
            self.adam_chunk_steps <= 0
            or self.adam_handoff_window <= 0
            or self.adam_handoff_patience <= 0
        ):
            raise ValueError(
                "Adam chunk, handoff window, and patience must be positive"
            )
        if not all(
            isinstance(value, Real)
            and not isinstance(value, bool)
            and np.isfinite(value)
            for value in (
                self.lbfgs_ftol,
                self.adam_handoff_tolerance_per_site,
            )
        ):
            raise ValueError("optimizer tolerances must be finite real scalars")
        if (
            self.lbfgs_ftol < 0
            or self.adam_handoff_tolerance_per_site < 0
        ):
            raise ValueError("optimizer tolerances must be nonnegative")
        if (
            isinstance(self.lbfgs_gtol_per_site, bool)
            or not isinstance(self.lbfgs_gtol_per_site, Real)
            or not np.isfinite(self.lbfgs_gtol_per_site)
            or self.lbfgs_gtol_per_site < 0
        ):
            raise ValueError(
                "lbfgs_gtol_per_site must be a finite nonnegative scalar"
            )


@dataclass(frozen=True)
class ThermodynamicMetrics:
    """Host-side total F, total E, entropy in nats, norm, and minimum weight.

    The final weight is the backend's normalized structural-spectrum minimum,
    not a guarantee of full Hilbert-space rank or positivity before clipping.
    """

    free_energy: float
    energy: float
    entropy_nats: float
    norm: float
    minimum_structural_weight: float


@dataclass(frozen=True)
class OptimizationTracePoint:
    """One evaluated point with physical metrics and optimizer diagnostics.

    Gradient norms and objective values refer to ``C = beta*F/N``. Stored
    free energies are unscaled totals; elapsed times exclude compilation.
    Adam points in the same compiled chunk share that chunk's completion time.
    """

    phase: str
    call_index: int
    metrics: ThermodynamicMetrics
    gradient_norm: float | None
    best_free_energy: float
    phase_call_index: int = 0
    gradient_norm_inf: float | None = None
    objective_value: float | None = None
    best_objective_value: float | None = None
    elapsed_seconds: float = 0.0
    phase_elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class AdamLBFGSBResult:
    """Best evaluated endpoint, phase endpoints, and optimization telemetry.

    ``parameters`` and the top-level thermodynamic fields describe the best
    evaluated point, not necessarily the terminal SciPy iterate. Parameter
    arrays are read-only copies. ``criteria_converged`` requires both the Adam
    handoff criterion and native L-BFGS-B convergence; retained safety-limit
    exits do not satisfy it. Runtime fields exclude the separately recorded
    compilation time.
    """

    parameters: np.ndarray
    free_energy: float
    energy: float
    entropy_nats: float
    norm: float
    minimum_structural_weight: float
    initial_parameters: np.ndarray
    adam_best_parameters: np.ndarray
    adam_parameters: np.ndarray
    terminal_parameters: np.ndarray
    initial_metrics: ThermodynamicMetrics
    adam_best_metrics: ThermodynamicMetrics
    adam_terminal_metrics: ThermodynamicMetrics
    terminal_metrics: ThermodynamicMetrics
    adam_best_call_index: int
    best_phase: str
    best_call_index: int
    num_evaluations: int
    adam_evaluations: int
    adam_chunks: int
    adam_stop_reason: str
    adam_handoff_improvement_per_site: float | None
    adam_handoff_objective_gain: float | None
    adam_handoff_consecutive_checks: int
    termination_mode: str
    adam_watchdog_steps: int
    adam_watchdog_triggered: bool
    lbfgs_evaluations: int
    lbfgs_iterations: int
    lbfgs_watchdog_triggered: bool
    lbfgs_stop_reason: str
    lbfgs_start_source: str
    criteria_converged: bool
    effective_lbfgs_gtol: float
    effective_lbfgs_maxfun: int
    objective_name: str
    objective_scale: float
    objective_normalized_by_system_size: bool
    objective_dimensionless: bool
    objective_value: float
    terminal_projected_gradient_inf: float | None
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    compile_seconds: float
    compile_cache_hit: bool
    compile_cache_key: str
    adam_seconds: float
    lbfgs_seconds: float
    total_seconds: float
    trace: tuple[OptimizationTracePoint, ...]
    adam_watchdog_action: str = "raise"
    lbfgs_nonconvergence_action: str = "raise"
    lbfgs_nfev: int = 0
    lbfgs_njev: int = 0

    @property
    def internal_metrics(self) -> ThermodynamicMetrics:
        """Return the best endpoint's internally computed physical metrics."""
        return ThermodynamicMetrics(
            self.free_energy,
            self.energy,
            self.entropy_nats,
            self.norm,
            self.minimum_structural_weight,
        )


_COMPILED_OPTIMIZER_CACHE_MAXSIZE = 8
_COMPILED_OPTIMIZER_CACHE: OrderedDict[tuple[Any, ...], Any] = (
    OrderedDict()
)


def _readonly_array(parameters: Sequence[float]) -> np.ndarray:
    """Copy parameters to float64 and disable writes on the returned array."""
    result = np.array(parameters, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _metrics(values: Sequence[Any]) -> ThermodynamicMetrics:
    """Convert five ordered backend scalar values to host-side metrics."""
    converted = tuple(float(np.asarray(value)) for value in values)
    return ThermodynamicMetrics(*converted)


def _classify_lbfgs_stop(
    *, success: bool, status: int, message: str
) -> str | None:
    """Return ``ftol`` or ``gtol`` for native SciPy convergence, else None."""

    if not success or status != 0:
        return None
    normalized = message.upper().replace(" ", "_").replace("-", "_")
    if "PROJECTED_GRADIENT" in normalized or "PGTOL" in normalized:
        return "gtol"
    if (
        "REL_REDUCTION_OF_F" in normalized
        or "FACTR" in normalized
        or "FTOL" in normalized
    ):
        return "ftol"
    return None


def optimize_adam_lbfgsb(
    spec: Any,
    initial_params: Sequence[float],
    options: AdamLBFGSBOptions | None = None,
) -> AdamLBFGSBResult:
    """Optimize one start using criterion-driven Adam followed by L-BFGS-B.

    By default, criterion-driven optimization raises
    ``OptimizationConvergenceError`` if either phase reaches a watchdog
    or L-BFGS-B exits for any reason other than native ``ftol``/``gtol``
    convergence. Protocol-specific watchdog actions may retain an unconverged
    endpoint. Optimization always uses dimensionless free-energy density.

    Args:
        spec: Immutable, hashable backend specification with positive beta,
            system size, parameter count, and optional backend module name.
        initial_params: Flat parameter vector, clipped to the configured
            bounds before either optimizer phase.
        options: Validated stopping and bound settings, or None for defaults.

    Returns:
        Best evaluated parameters, physical total-F metrics, and phase traces.
        Trace objectives and gradients use ``C = beta*F/N``; backend losses
        remain ``F = E - S/beta`` with natural-log entropy.

    Raises:
        OptimizationConvergenceError: A disallowed safety limit or abnormal
            L-BFGS-B termination is encountered.
        FloatingPointError: An evaluated metric or gradient is nonfinite.
    """

    options = AdamLBFGSBOptions() if options is None else options
    if not isinstance(options, AdamLBFGSBOptions):
        raise TypeError("options must be AdamLBFGSBOptions")

    # Heavy imports remain local so launchers can set the thread environment
    # before initializing JAX or SciPy in spawned workers.
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import optax
    from scipy.optimize import Bounds, minimize

    backend_module_name = getattr(spec, "optimizer_backend_module", None)
    if backend_module_name is None:
        try:
            from .jax_mps_gibbs import make_loss_with_aux, thermodynamics
        except ImportError:  # Support historical top-level imports.
            from jax_mps_gibbs import make_loss_with_aux, thermodynamics
    else:
        if not isinstance(backend_module_name, str) or not backend_module_name:
            raise TypeError(
                "spec.optimizer_backend_module must be a non-empty module name"
            )
        backend = importlib.import_module(backend_module_name)
        try:
            make_loss_with_aux = backend.make_loss_with_aux
            thermodynamics = backend.thermodynamics
        except AttributeError as error:
            raise TypeError(
                f"optimizer backend {backend_module_name!r} must provide "
                "make_loss_with_aux and thermodynamics"
            ) from error

    initial = np.asarray(initial_params, dtype=np.float64)
    if initial.ndim != 1 or initial.size != int(spec.num_parameters):
        raise ValueError(
            f"expected {spec.num_parameters} parameters, found shape {initial.shape}"
        )
    if not np.isfinite(initial).all():
        raise ValueError("initial parameters must be finite")
    initial = np.clip(initial, options.lower_bound, options.upper_bound)

    adam = optax.adam(options.adam_learning_rate)
    num_system = int(spec.num_system)
    if num_system <= 0:
        raise ValueError("spec.num_system must be positive")
    beta = float(spec.beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("spec.beta must be finite and positive")
    adam_limit = options.adam_watchdog_steps
    objective_scale = beta / num_system
    objective_name = "dimensionless_free_energy_density"
    handoff_scale = objective_scale
    effective_lbfgs_gtol = float(options.lbfgs_gtol_per_site) * 1.0
    effective_lbfgs_maxfun = (
        (options.lbfgs_maxiter + 1) * (options.lbfgs_maxls + 1)
    )
    adam_execution_chunk = options.adam_chunk_steps
    chunk_sizes = {
        min(adam_execution_chunk, remaining)
        for remaining in range(adam_limit, 0, -adam_execution_chunk)
    }
    # Only values closed over by a lowered JAX program belong in this key.
    # Python-side handoff rules and SciPy stopping settings can therefore share
    # the same compiled executable while retaining fresh per-run state.
    cache_key = (
        spec,
        float(options.adam_learning_rate),
        float(options.lower_bound),
        float(options.upper_bound),
        tuple(sorted(chunk_sizes)),
    )
    try:
        hash(cache_key)
    except TypeError as error:
        raise TypeError(
            "spec must be an immutable, hashable optimizer specification "
            "for process-local compile reuse"
        ) from error
    compile_cache_key = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()
    cached_bundle = _COMPILED_OPTIMIZER_CACHE.get(cache_key)
    compile_cache_hit = cached_bundle is not None
    if compile_cache_hit:
        _COMPILED_OPTIMIZER_CACHE.move_to_end(cache_key)

    raw_value_and_grad = None

    def make_adam_chunk_program(chunk_steps: int):
        """Build one fixed-size chunk so repeated checks do not recompile."""

        def adam_chunk_program(
            params,
            state,
            best_loss,
            best_params,
            best_aux,
            best_call,
            valid_so_far,
            call_offset,
        ):
            """Run a fixed-size Adam scan and return its carry and history.

            Best-point tracking uses evaluated pre-update parameters; the
            final carried parameters include the last proposed update.
            """
            assert raw_value_and_grad is not None
            carry = (
                params,
                state,
                best_loss,
                best_params,
                best_aux,
                best_call,
                valid_so_far,
            )

            def step(step_carry, local_step_index):
                """Evaluate one point, track the best, and propose a bounded step.

                Nonfinite values freeze the parameter and optimizer-state
                updates and mark the chunk invalid for the host-side check.
                """
                (
                    step_params,
                    step_state,
                    step_best_loss,
                    step_best_params,
                    step_best_aux,
                    step_best_call,
                    step_valid_so_far,
                ) = step_carry
                (loss, aux), gradient = raw_value_and_grad(step_params)
                finite = jnp.logical_and(
                    jnp.logical_and(
                        jnp.isfinite(loss), jnp.all(jnp.isfinite(gradient))
                    ),
                    jnp.all(
                        jnp.stack(tuple(jnp.isfinite(value) for value in aux))
                    ),
                )
                valid = jnp.logical_and(step_valid_so_far, finite)
                better = jnp.logical_and(finite, loss < step_best_loss)
                next_best_loss = jnp.where(better, loss, step_best_loss)
                next_best_params = jnp.where(
                    better, step_params, step_best_params
                )
                next_best_aux = jax.tree_util.tree_map(
                    lambda candidate, incumbent: jnp.where(
                        better, candidate, incumbent
                    ),
                    aux,
                    step_best_aux,
                )
                global_call = call_offset + local_step_index + 1
                next_best_call = jnp.where(
                    better, global_call, step_best_call
                )
                optimizer_gradient = gradient * objective_scale
                updates, proposed_state = adam.update(
                    optimizer_gradient, step_state, step_params
                )
                proposed_params = jnp.clip(
                    optax.apply_updates(step_params, updates),
                    options.lower_bound,
                    options.upper_bound,
                )
                next_params = jnp.where(valid, proposed_params, step_params)
                next_state = jax.tree_util.tree_map(
                    lambda proposed, current: jnp.where(
                        valid, proposed, current
                    ),
                    proposed_state,
                    step_state,
                )
                output = (
                    loss,
                    *aux,
                    jnp.linalg.norm(optimizer_gradient),
                    jnp.max(jnp.abs(optimizer_gradient)),
                    next_best_loss,
                    finite,
                )
                return (
                    next_params,
                    next_state,
                    next_best_loss,
                    next_best_params,
                    next_best_aux,
                    next_best_call,
                    valid,
                ), output

            final_carry, history = jax.lax.scan(
                step,
                carry,
                jnp.arange(chunk_steps, dtype=jnp.int32),
                unroll=1,
            )
            return (*final_carry, history)

        return jax.jit(adam_chunk_program)

    params_device = jnp.asarray(initial, dtype=jnp.float64)
    adam_state_device = adam.init(params_device)
    if cached_bundle is None:
        compile_start = time.perf_counter()
        loss_with_aux = make_loss_with_aux(spec)
        raw_value_and_grad = jax.value_and_grad(loss_with_aux, has_aux=True)
        metrics_fn = jax.jit(lambda params: thermodynamics(params, spec))
        value_and_grad_fn = jax.jit(raw_value_and_grad)
        compiled_metrics = metrics_fn.lower(params_device).compile()
        compiled_value_and_grad = value_and_grad_fn.lower(params_device).compile()
        # The actual first point supplies the auxiliary shapes used when
        # lowering Adam. This compile probe is not an optimizer evaluation.
        initial_values_device = compiled_metrics(params_device)
        initial_aux = tuple(initial_values_device[1:])
        compiled_adam_chunks = {}
        for chunk_steps in sorted(chunk_sizes):
            adam_chunk_fn = make_adam_chunk_program(chunk_steps)
            compiled_adam_chunks[chunk_steps] = adam_chunk_fn.lower(
                params_device,
                adam_state_device,
                initial_values_device[0],
                params_device,
                initial_aux,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
            ).compile()
        compile_seconds = time.perf_counter() - compile_start
        cached_bundle = (
            compiled_metrics,
            compiled_value_and_grad,
            compiled_adam_chunks,
        )
        _COMPILED_OPTIMIZER_CACHE[cache_key] = cached_bundle
        _COMPILED_OPTIMIZER_CACHE.move_to_end(cache_key)
        while len(_COMPILED_OPTIMIZER_CACHE) > _COMPILED_OPTIMIZER_CACHE_MAXSIZE:
            _COMPILED_OPTIMIZER_CACHE.popitem(last=False)
    else:
        compile_seconds = 0.0
        (
            compiled_metrics,
            compiled_value_and_grad,
            compiled_adam_chunks,
        ) = cached_bundle
        initial_values_device = compiled_metrics(params_device)
        initial_aux = tuple(initial_values_device[1:])

    initial_values = _metrics(jax.device_get(initial_values_device))

    total_start = time.perf_counter()
    trace: list[OptimizationTracePoint] = [
        OptimizationTracePoint(
            phase="initial",
            call_index=0,
            metrics=initial_values,
            gradient_norm=None,
            best_free_energy=initial_values.free_energy,
            phase_call_index=0,
            gradient_norm_inf=None,
            objective_value=initial_values.free_energy * objective_scale,
            best_objective_value=initial_values.free_energy * objective_scale,
            elapsed_seconds=0.0,
            phase_elapsed_seconds=0.0,
        )
    ]
    adam_start = time.perf_counter()
    best_loss_device = initial_values_device[0]
    best_parameters_device = params_device
    best_aux_device = initial_aux
    best_call_device = jnp.asarray(0, dtype=jnp.int32)
    valid_device = jnp.asarray(True)
    adam_steps_used = 0
    adam_chunks = 0
    adam_handoff_improvement_per_site: float | None = None
    adam_handoff_objective_gain: float | None = None
    adam_handoff_consecutive_checks = 0
    adam_stop_reason = "watchdog_exhausted"
    running_best = [initial_values.free_energy]
    effective_min_steps = options.adam_min_steps

    while adam_steps_used < adam_limit:
        chunk_steps = min(
            adam_execution_chunk, adam_limit - adam_steps_used
        )
        adam_output = jax.device_get(
            compiled_adam_chunks[chunk_steps](
                params_device,
                adam_state_device,
                best_loss_device,
                best_parameters_device,
                best_aux_device,
                best_call_device,
                valid_device,
                jnp.asarray(adam_steps_used, dtype=jnp.int32),
            )
        )
        (
            params_device,
            adam_state_device,
            best_loss_device,
            best_parameters_device,
            best_aux_device,
            best_call_device,
            valid_device,
            history,
        ) = adam_output
        chunk_elapsed_seconds = time.perf_counter() - total_start
        chunk_phase_elapsed_seconds = time.perf_counter() - adam_start
        finite_history = np.asarray(history[-1], dtype=bool)
        if not bool(valid_device):
            first = int(np.flatnonzero(~finite_history)[0]) + adam_steps_used + 1
            raise FloatingPointError(
                f"nonfinite Adam value or gradient at step {first}"
            )

        for local_index in range(chunk_steps):
            call_index = adam_steps_used + local_index + 1
            point_metrics = ThermodynamicMetrics(
                *(
                    float(np.asarray(history[field][local_index]))
                    for field in range(5)
                )
            )
            point_best = float(np.asarray(history[7][local_index]))
            running_best.append(point_best)
            trace.append(
                OptimizationTracePoint(
                    phase="adam",
                    call_index=call_index,
                    metrics=point_metrics,
                    gradient_norm=float(
                        np.asarray(history[5][local_index])
                    ),
                    best_free_energy=point_best,
                    phase_call_index=call_index,
                    gradient_norm_inf=float(
                        np.asarray(history[6][local_index])
                    ),
                    objective_value=(
                        point_metrics.free_energy * objective_scale
                    ),
                    best_objective_value=point_best * objective_scale,
                    elapsed_seconds=chunk_elapsed_seconds,
                    phase_elapsed_seconds=chunk_phase_elapsed_seconds,
                )
            )

        adam_steps_used += chunk_steps
        adam_chunks += 1
        if (
            adam_steps_used >= effective_min_steps
            and adam_steps_used >= options.adam_handoff_window
        ):
            earlier = running_best[
                adam_steps_used - options.adam_handoff_window
            ]
            current = running_best[adam_steps_used]
            adam_handoff_objective_gain = (earlier - current) * handoff_scale
            # Retain the historical field for serialized-result compatibility;
            # for the dimensionless objective it is exactly Delta C.
            adam_handoff_improvement_per_site = adam_handoff_objective_gain
            if (
                adam_handoff_improvement_per_site
                <= options.adam_handoff_tolerance_per_site
            ):
                adam_handoff_consecutive_checks += 1
            else:
                adam_handoff_consecutive_checks = 0
            if (
                adam_handoff_consecutive_checks
                >= options.adam_handoff_patience
            ):
                adam_stop_reason = "handoff_tolerance"
                break

    adam_seconds = time.perf_counter() - adam_start
    adam_watchdog_triggered = adam_stop_reason == "watchdog_exhausted"
    if adam_watchdog_triggered and options.adam_watchdog_action == "raise":
        raise OptimizationConvergenceError(
            "Adam did not satisfy the handoff criterion before the "
            f"{options.adam_watchdog_steps}-step safety watchdog",
            phase="adam",
            stop_reason=adam_stop_reason,
            telemetry={
                "termination_mode": "criteria",
                "adam_evaluations": adam_steps_used,
                "adam_chunks": adam_chunks,
                "adam_watchdog_steps": options.adam_watchdog_steps,
                "adam_watchdog_triggered": True,
                "adam_handoff_objective_gain": (
                    adam_handoff_objective_gain
                ),
                "adam_handoff_consecutive_checks": (
                    adam_handoff_consecutive_checks
                ),
                "adam_seconds": adam_seconds,
                "objective_name": objective_name,
                "objective_scale": objective_scale,
                "best_free_energy": float(np.asarray(best_loss_device)),
            },
        )

    best_metrics = ThermodynamicMetrics(
        float(np.asarray(best_loss_device)),
        *(float(np.asarray(value)) for value in best_aux_device),
    )
    best_params = np.asarray(best_parameters_device, dtype=np.float64).copy()
    best_phase = "initial" if int(best_call_device) == 0 else "adam"
    best_call_index = int(best_call_device)
    adam_best_params = best_params.copy()
    adam_best_metrics = best_metrics
    adam_best_call_index = best_call_index
    adam_params = np.asarray(params_device, dtype=np.float64)
    adam_terminal_metrics = _metrics(
        jax.device_get(compiled_metrics(jnp.asarray(adam_params)))
    )

    lbfgs_calls = 0
    lbfgs_start = time.perf_counter()
    lbfgs_initial_params = adam_best_params.copy()
    lbfgs_start_source = "adam_best"

    def scipy_objective(params_numpy: np.ndarray):
        """Return C and its gradient while recording the L-BFGS-B evaluation.

        Update the shared best-point record using total F, which has the same
        ordering as C at the fixed positive beta and system size.
        """
        nonlocal lbfgs_calls, best_metrics, best_params, best_phase, best_call_index
        (loss_and_aux, gradient) = jax.device_get(
            compiled_value_and_grad(jnp.asarray(params_numpy, dtype=jnp.float64))
        )
        loss, aux = loss_and_aux
        gradient_array = np.asarray(gradient, dtype=np.float64)
        values = ThermodynamicMetrics(
            float(np.asarray(loss)),
            *(float(np.asarray(value)) for value in aux),
        )
        if not np.isfinite(gradient_array).all() or not all(
            np.isfinite(value)
            for value in (
                values.free_energy,
                values.energy,
                values.entropy_nats,
                values.norm,
                values.minimum_structural_weight,
            )
        ):
            raise FloatingPointError("nonfinite L-BFGS-B value or gradient")
        lbfgs_calls += 1
        call_index = adam_steps_used + lbfgs_calls
        if values.free_energy < best_metrics.free_energy:
            best_metrics = values
            best_params = np.asarray(params_numpy, dtype=np.float64).copy()
            best_phase = "lbfgs"
            best_call_index = call_index
        trace.append(
            OptimizationTracePoint(
                phase="lbfgs",
                call_index=call_index,
                metrics=values,
                gradient_norm=float(
                    np.linalg.norm(gradient_array * objective_scale)
                ),
                best_free_energy=best_metrics.free_energy,
                phase_call_index=lbfgs_calls,
                gradient_norm_inf=float(
                    np.max(np.abs(gradient_array * objective_scale))
                ),
                objective_value=values.free_energy * objective_scale,
                best_objective_value=(
                    best_metrics.free_energy * objective_scale
                ),
                elapsed_seconds=time.perf_counter() - total_start,
                phase_elapsed_seconds=time.perf_counter() - lbfgs_start,
            )
        )
        return (
            values.free_energy * objective_scale,
            gradient_array * objective_scale,
        )

    scipy_options = {
        "maxiter": options.lbfgs_maxiter,
        "maxls": options.lbfgs_maxls,
        "ftol": options.lbfgs_ftol,
        "gtol": effective_lbfgs_gtol,
        "disp": False,
    }
    # SciPy otherwise applies a hidden 15,000-evaluation cap. With an
    # analytic Jacobian, this bound lets the explicit iteration watchdog
    # and per-iteration line-search limit control the run.
    scipy_options["maxfun"] = effective_lbfgs_maxfun
    result = minimize(
        scipy_objective,
        x0=lbfgs_initial_params,
        method="L-BFGS-B",
        jac=True,
        bounds=Bounds(options.lower_bound, options.upper_bound),
        options=scipy_options,
    )
    lbfgs_seconds = time.perf_counter() - lbfgs_start
    terminal_params = np.asarray(result.x, dtype=np.float64)
    lbfgs_iterations = int(result.nit)
    lbfgs_nfev = int(getattr(result, "nfev", lbfgs_calls))
    lbfgs_njev = int(getattr(result, "njev", lbfgs_calls))
    optimizer_success = bool(result.success)
    optimizer_status = int(result.status)
    optimizer_message = str(result.message)
    terminal_gradient = np.asarray(result.jac, dtype=np.float64).copy()
    if (
        not np.isfinite(terminal_params).all()
        or not np.isfinite(terminal_gradient).all()
    ):
        raise FloatingPointError(
            "nonfinite L-BFGS-B terminal parameters or gradient"
        )
    at_lower_and_outward = np.logical_and(
        terminal_params <= options.lower_bound, terminal_gradient > 0.0
    )
    at_upper_and_outward = np.logical_and(
        terminal_params >= options.upper_bound, terminal_gradient < 0.0
    )
    terminal_gradient[
        np.logical_or(at_lower_and_outward, at_upper_and_outward)
    ] = 0.0
    terminal_projected_gradient_inf = float(
        np.max(np.abs(terminal_gradient), initial=0.0)
    )
    native_stop = _classify_lbfgs_stop(
        success=optimizer_success,
        status=optimizer_status,
        message=optimizer_message,
    )
    normalized_message = optimizer_message.upper().replace(" ", "_")
    if native_stop is not None:
        lbfgs_stop_reason = native_stop
    elif "ITERATIONS_REACHED_LIMIT" in normalized_message:
        lbfgs_stop_reason = "watchdog_exhausted"
    elif (
        "EVALUATIONS_EXCEEDS_LIMIT" in normalized_message
        or "EVALUATION_LIMIT" in normalized_message
    ):
        lbfgs_stop_reason = "evaluation_watchdog_exhausted"
    elif optimizer_status == 1 and lbfgs_iterations >= options.lbfgs_maxiter:
        lbfgs_stop_reason = "watchdog_exhausted"
    elif optimizer_status == 1:
        lbfgs_stop_reason = "evaluation_watchdog_exhausted"
    else:
        lbfgs_stop_reason = "abnormal_termination"
    lbfgs_watchdog_triggered = lbfgs_stop_reason in {
        "watchdog_exhausted",
        "evaluation_watchdog_exhausted",
    }
    if (
        native_stop is None
        and (
            options.lbfgs_nonconvergence_action == "raise"
            or not lbfgs_watchdog_triggered
        )
    ):
        raise OptimizationConvergenceError(
            "L-BFGS-B did not terminate through native ftol or gtol "
            f"convergence: {optimizer_message}",
            phase="lbfgs",
            stop_reason=lbfgs_stop_reason,
            telemetry={
                "termination_mode": "criteria",
                "adam_evaluations": adam_steps_used,
                "adam_stop_reason": adam_stop_reason,
                "lbfgs_evaluations": lbfgs_calls,
                "lbfgs_iterations": lbfgs_iterations,
                "lbfgs_maxiter": options.lbfgs_maxiter,
                "effective_lbfgs_maxfun": effective_lbfgs_maxfun,
                "lbfgs_watchdog_triggered": (
                    lbfgs_watchdog_triggered
                ),
                "lbfgs_stop_reason": lbfgs_stop_reason,
                "lbfgs_start_source": lbfgs_start_source,
                "optimizer_success": optimizer_success,
                "optimizer_status": optimizer_status,
                "optimizer_message": optimizer_message,
                "terminal_projected_gradient_inf": (
                    terminal_projected_gradient_inf
                ),
                "best_free_energy": best_metrics.free_energy,
                "lbfgs_seconds": lbfgs_seconds,
                "objective_name": objective_name,
                "objective_scale": objective_scale,
            },
        )

    terminal_metrics = _metrics(
        jax.device_get(compiled_metrics(jnp.asarray(terminal_params)))
    )
    if not all(
        np.isfinite(value)
        for value in (
            terminal_metrics.free_energy,
            terminal_metrics.energy,
            terminal_metrics.entropy_nats,
            terminal_metrics.norm,
            terminal_metrics.minimum_structural_weight,
        )
    ):
        raise FloatingPointError("nonfinite L-BFGS-B terminal metrics")
    total_seconds = time.perf_counter() - total_start
    frozen_best = _readonly_array(best_params)
    return AdamLBFGSBResult(
        parameters=frozen_best,
        free_energy=best_metrics.free_energy,
        energy=best_metrics.energy,
        entropy_nats=best_metrics.entropy_nats,
        norm=best_metrics.norm,
        minimum_structural_weight=best_metrics.minimum_structural_weight,
        initial_parameters=_readonly_array(initial),
        adam_best_parameters=_readonly_array(adam_best_params),
        adam_parameters=_readonly_array(adam_params),
        terminal_parameters=_readonly_array(terminal_params),
        initial_metrics=initial_values,
        adam_best_metrics=adam_best_metrics,
        adam_terminal_metrics=adam_terminal_metrics,
        terminal_metrics=terminal_metrics,
        adam_best_call_index=adam_best_call_index,
        best_phase=best_phase,
        best_call_index=best_call_index,
        num_evaluations=adam_steps_used + lbfgs_calls,
        adam_evaluations=adam_steps_used,
        adam_chunks=adam_chunks,
        adam_stop_reason=adam_stop_reason,
        adam_handoff_improvement_per_site=adam_handoff_improvement_per_site,
        adam_handoff_objective_gain=adam_handoff_objective_gain,
        adam_handoff_consecutive_checks=adam_handoff_consecutive_checks,
        termination_mode="criteria",
        adam_watchdog_steps=options.adam_watchdog_steps,
        adam_watchdog_triggered=adam_watchdog_triggered,
        adam_watchdog_action=options.adam_watchdog_action,
        lbfgs_evaluations=lbfgs_calls,
        lbfgs_iterations=lbfgs_iterations,
        lbfgs_nfev=lbfgs_nfev,
        lbfgs_njev=lbfgs_njev,
        lbfgs_watchdog_triggered=lbfgs_watchdog_triggered,
        lbfgs_nonconvergence_action=options.lbfgs_nonconvergence_action,
        lbfgs_stop_reason=lbfgs_stop_reason,
        lbfgs_start_source=lbfgs_start_source,
        criteria_converged=(
            adam_stop_reason == "handoff_tolerance"
            and native_stop is not None
        ),
        effective_lbfgs_gtol=effective_lbfgs_gtol,
        effective_lbfgs_maxfun=effective_lbfgs_maxfun,
        objective_name=objective_name,
        objective_scale=objective_scale,
        objective_normalized_by_system_size=True,
        objective_dimensionless=True,
        objective_value=best_metrics.free_energy * objective_scale,
        terminal_projected_gradient_inf=terminal_projected_gradient_inf,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        compile_seconds=compile_seconds,
        compile_cache_hit=compile_cache_hit,
        compile_cache_key=compile_cache_key,
        adam_seconds=adam_seconds,
        lbfgs_seconds=lbfgs_seconds,
        total_seconds=total_seconds,
        trace=tuple(trace),
    )


__all__ = [
    "AdamLBFGSBOptions",
    "AdamLBFGSBResult",
    "OptimizationConvergenceError",
    "OptimizationTracePoint",
    "ThermodynamicMetrics",
    "optimize_adam_lbfgsb",
]
