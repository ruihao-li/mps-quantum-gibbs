# MPS-assisted variational Gibbs-state preparation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python: tested on 3.12](https://img.shields.io/badge/Python-tested_on_3.12-blue.svg)](https://www.python.org/downloads/) [![JAX](https://img.shields.io/badge/JAX-green.svg)](https://docs.jax.dev/) [![Quimb](https://img.shields.io/badge/Quimb-green.svg)](https://quimb.readthedocs.io/) [![Qiskit](https://img.shields.io/badge/Qiskit-green.svg)](https://quantum.cloud.ibm.com/docs/en/api/qiskit) [![arXiv](https://img.shields.io/badge/arXiv-2510.23546-b31b1b.svg)](https://doi.org/10.48550/arXiv.2510.23546)

Numerical code for optimizing purification circuits (TFDA, contiguous and interleaved HEAs), computing thermal observables and exact references, and analyzing optimization and gradient statistics, accompanying [arXiv:2510.23546](https://doi.org/10.48550/arXiv.2510.23546).

## Setup

Run from the repository root. Computation is CPU-based with float64 precision (complex128 where needed by the small-system statevector backend):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned versions were exercised on macOS arm64. Clean installation and Linux/HPC execution remain untested. Native Windows is unsupported by the Unix resource monitor.

## Layout

| Directory | Purpose |
|---|---|
| [`src/`](src/) | Contiguous/interleaved HEA and TFDA circuits, lattices, Hamiltonians, JAX objectives, and Adam–L-BFGS-B |
| [`experiments/`](experiments/) | Configurations, seeds, launchers, validation, and independent endpoint scoring |
| [`references/`](references/) | Exact thermal/ground-state solvers, independent validation, and shared checksum |
| [`analysis/`](analysis/) | Observable reconstruction, gradient statistics, and CSV summaries |

Optimization uses the JAX exact-MPS backend for the HEA studies and an exact statevector backend for the small TFDA/HEA benchmark. Independent endpoint checks use Qiskit/Quimb MPS reconstruction for the former and Qiskit dense states for the latter. The launchers below are the supported workflow entry points. Run the scripts directly.

In `analysis/`, `observables.py` computes moments and connected correlations from MPSs; `gradient_statistics.py` analyzes saved energy/entropy gradients and bootstrap intervals; `export_tables.py` checks completed runs, selects endpoints, and writes summary tables.

## Numerical protocols

HEA denotes the hardware-efficient ansatz and TFDA the thermofield-double ansatz. `N` is the number of system spins, `Na` the number of ancillas, and `L` the number of circuit layers.

| Study | Settings | Work |
|---|---|---|
| HEA | 10×1, 20×1, 3×3, 4×4; seven ansatz configurations; eight temperatures; 20 cold starts | 4,480 optimizations |
| Small benchmark | TFIM/XXZ; N=4,6; TFDA/contiguous HEA; ten temperatures; 20 cold starts | 1,600 optimizations |
| Gradient scaling | N=10,12,16,20,24,30; contiguous (8,7), interleaved (10,3); 128 vectors per shape | 1,536 gradient samples |

HEA contiguous configurations are `(Na,L)=(4,3),(6,5),(6,7),(8,7)`. Interleaved configurations are `(9,2),(10,2),(10,3)` for 10×1; `(8,2),(9,2),(9,3)` for 3×3; and `(10,2),(11,2),(10,3)` for 20×1 and 4×4. The 1D inverse temperature grid is `0.1,0.4,1,1.1,1.2,1.8,2.1,3`; the 2D grid is `0.05,0.2,0.4,0.45,0.5,0.7,0.85,2`. The small benchmark uses ten equally spaced beta values from 0.1 to 2.8, with Na=N and L=N/2. [experiments/protocols.py](experiments/protocols.py) specifies the complete grids, seeds, and stopping policies.

Entropy uses natural logarithms and optimization minimizes `C_beta=(beta*E-S)/N`. The HEA TFIM has open boundaries and `H=-sum_edges ZiZj-0.5*sum_i Xi`. Quimb independently reconstructs the best and terminal endpoints with zero cutoff and bond caps `2**L` (contiguous) or `4**L` (interleaved). Analysis selects the lowest recomputed C among 20 starts, never the state with the best observable error.

Adam uses learning rate `0.02`, at least 200 steps, and 25-step JIT chunks. It checks the 50-step running-best gain in `C_beta` every 25 steps, handing its best evaluated parameters to L-BFGS-B when this gain is at most `1e-4` for the required number of consecutive checks below. SciPy-native stopping uses `ftol=2.220446049250313e-9` and `gtol=5e-7`, with safety limits `maxiter=1500`, `maxls=20`, and `maxfun=31521`.

| Protocol | Required consecutive low-improvement checks | Adam safety limit | Safety-limit handling |
|---|---:|---:|---|
| HEA | 1 | 2,000 steps | Hand off at the Adam limit; retain and flag endpoints at L-BFGS-B limits |
| Small benchmark | 2 | 5,000 steps | Raise on an Adam limit or L-BFGS-B nonconvergence |


## Validate, then run

Run these commands from this directory. Validation uses small generated problems, independent reconstructions, and directional derivatives—not full production runs. Built-in checks remain mandatory even though the separate development test suite is not distributed here.

```sh
python -B experiments/run.py hea --output-root ../gibbs-results --plan
python -B experiments/run.py small --output-root ../gibbs-results --plan
python -B experiments/gradient.py --output-root ../gibbs-results --plan
python -B experiments/run.py hea --output-root ../gibbs-results --validate
python -B experiments/run.py small --output-root ../gibbs-results --validate
python -B experiments/gradient.py --output-root ../gibbs-results --validate
python -B references/run_references.py --data-root ../gibbs-results --validate-only
```

`--plan` prints configurations without running the numerical study. Validation writes its reports to the chosen output root; reference validation also generates the small 3×3 exact reference used by its independent checks. Experiment launchers refuse missing or mismatched validation.

The following commands launch the **expensive full calculations**:

```sh
python -B experiments/run.py hea --output-root ../gibbs-results --launch --workers 1
python -B experiments/run.py small --output-root ../gibbs-results --launch --workers 1
python -B experiments/gradient.py --output-root ../gibbs-results --launch --workers 1
python -B references/run_references.py --data-root ../gibbs-results --threads 1
```

Progress is printed in the terminal. Default one-worker execution is conservative; qualify memory and throughput before increasing `--workers`. Each optimizer worker sets numerical-thread limits to one and reuses compiled kernels. `--threads` controls the separate exact-reference solver. References use exact open-chain covariance calculations, dense 3×3 diagonalization, and the complete symmetry-resolved 4×4 spectrum, with finite-size even-parity pure-ground-state comparators. The default reference command includes all four systems.

Rerun the same launch command to resume verified completed checkpoints. Do not run concurrent writers into the same campaign. Source, environment, options, and seed bindings prevent mixing incompatible runs; use a fresh output root after changing them. For a reference subset, use e.g. `--systems 10x1 20x1 3x3` with a separate data root.

## Analyze saved results

After the required campaigns and references are complete:

```sh
python -B analysis/export_tables.py --data-root ../gibbs-results --tables-root ../gibbs-tables
```

Use `--stage hea`, `--stage small`, or `--stage gradient` to process one complete study. The HEA stage requires all four exact reference systems; the small and gradient stages use their own saved campaign outputs and do not require that reference bundle. Partial or mismatched datasets are rejected. HEA analysis reconstructs only the 224 selected endpoints for full observables, while retaining all starts for optimization/time summaries. It computes `chi=beta*Var(Mz)/N`, `cv=beta**2*Var(H)/N`, and connected distance-averaged correlations. Gradient analysis uses 5,000 paired whole-vector bootstrap resamples for 95% intervals.

Raw parameters, traces, metrics, and integrity manifests remain under `../gibbs-results/{hea,small,gradient}/`; exact references are under `../gibbs-results/references/`. Endpoint caches go under `analysis/` within that results root. CSV summaries and their manifests are written to `../gibbs-tables/`, without overwriting differing existing files.

## Citation

If you use this code, please cite the [associated paper](https://doi.org/10.48550/arXiv.2510.23546). The bibliographic details below follow the [arXiv record](https://arxiv.org/abs/2510.23546):

```bibtex
@misc{li2025variationalthermal,
  title         = {Variational Thermal State Preparation on Digital Quantum Processors Assisted by Matrix Product States},
  author        = {Rui-Hao Li and Semeon Valgushev and Khadijeh Najafi},
  year          = {2025},
  eprint        = {2510.23546},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2510.23546},
  url           = {https://doi.org/10.48550/arXiv.2510.23546}
}
```

## License

This code is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full terms.
