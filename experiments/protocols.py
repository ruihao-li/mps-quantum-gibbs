"""Final-paper configuration matrices and original deterministic seed mappings."""
from __future__ import annotations
from common import np
from gibbs_optimizers import AdamLBFGSBOptions
from jax_mps_gibbs import JAXMPSObjectiveSpec, chain_edges, square_edges
from jax_dense_gibbs import DenseGibbsObjectiveSpec

BETA_1D = (.1, .4, 1., 1.1, 1.2, 1.8, 2.1, 3.)
BETA_2D = (.05, .2, .4, .45, .5, .7, .85, 2.)
CONTIGUOUS = ((4, 3), (6, 5), (6, 7), (8, 7))


def hea_settings():
    """Return the 224 HEA settings with historical cold-start bank indices intact."""
    rows = []
    # Original contiguous bank ordering included N=30 and 6x6; DO NOT renumber.
    for nx, ny, old_system_index in ((10, 1, 0), (20, 1, 1), (3, 3, 3), (4, 4, 4)):
        n = nx * ny
        shapes = [("contiguous", na, layers, None, old_system_index * 4 + k)
                  for k, (na, layers) in enumerate(CONTIGUOUS)]
        if ny == 1 and nx == 10:
            shapes += [("interleaved_pair", 9, 2, 2026091501, 0),
                       ("interleaved_pair", 10, 2, 2026081001, 0),
                       ("interleaved_pair", 10, 3, 2026081001, 1)]
        elif nx == ny == 3:
            shapes += [("interleaved_pair", 8, 2, 2026091501, 1),
                       ("interleaved_pair", 9, 2, 2026081001, 2),
                       ("interleaved_pair", 9, 3, 2026081001, 3)]
        else:
            offset = 0 if ny == 1 else 2
            shapes += [("interleaved_pair", 10, 2, 2026081102, offset),
                       ("interleaved_pair", 11, 2, 2026091404, offset // 2),
                       ("interleaved_pair", 10, 3, 2026081102, offset + 1)]
        for family, na, layers, namespace, bank in shapes:
            candidate = f"{family}_na{na}_L{layers}"
            for beta in BETA_1D if ny == 1 else BETA_2D:
                token = format(beta, ".12g").replace(".", "p")
                rows.append(dict(setting_id=f"{nx}x{ny}_{candidate}_beta{token}",
                    system=f"{nx}x{ny}", nx=nx, ny=ny, num_system=n,
                    num_ancillas=na, num_layers=layers, ansatz_type=family,
                    candidate=candidate, model="tfim", beta=beta, j=-1., hx=-.5, hz=0.,
                    seed_namespace=namespace, seed_bank_index=bank))
    return rows


def small_settings():
    """Return 80 TFIM/XXZ, TFDA/HEA settings for four- and six-spin comparisons."""
    rows = []
    for model in ("tfim", "xxz"):
        for n in (4, 6):
            for family in ("tfda", "hea"):
                for b, beta in enumerate((.1, .4, .7, 1., 1.3, 1.6, 1.9, 2.2, 2.5, 2.8)):
                    rows.append(dict(setting_id=f"{model}_{family}_N{n}_beta{b:02d}",
                        system=f"{n}x1", nx=n, ny=1, num_system=n,
                        num_ancillas=n, num_layers=n//2, ansatz_type=family,
                        candidate=family, model=model, beta=beta, beta_index=b,
                        j=-1., hx=-.5, hz=0., jxy=-1., jz=1.5))
    return rows


def gradient_shapes():
    """Return 12 fixed-depth chain architectures and their gradient seed namespaces."""
    rows = []
    for family, na, layers, namespace in (("contiguous", 8, 7, 101),
                                         ("interleaved_pair", 10, 3, 202)):
        for n in (10, 12, 16, 20, 24, 30):
            identity = f"{family}_N{n}_na{na}_L{layers}"
            rows.append(dict(shape_id=identity, setting_id=identity, system=f"{n}x1",
                nx=n, ny=1, num_system=n, num_ancillas=na, num_layers=layers,
                ansatz_type=family, beta=1., j=-1., hx=-.5, hz=0.,
                seed_namespace=namespace))
    return rows


def make_spec(setting, experiment="hea"):
    """Construct the dense-small or MPS objective specification for a campaign setting.

    Use open chain edges for ny = 1 and square edges otherwise. The release
    campaign matrices contain only chains and square lattices.
    """
    s = setting
    kwargs = dict(name=s["setting_id"], num_system=s["num_system"],
        num_ancillas=s["num_ancillas"], num_layers=s["num_layers"], beta=s["beta"],
        ansatz_type=s["ansatz_type"],
        edges=chain_edges(s["num_system"]) if s["ny"] == 1 else square_edges(s["nx"]))
    if experiment == "small":
        return DenseGibbsObjectiveSpec(model=s["model"], **kwargs)
    return JAXMPSObjectiveSpec(j=s["j"], hx=s["hx"], hz=s.get("hz", 0.), **kwargs)


def options(experiment):
    """Return the experiment's Adam-to-L-BFGS-B stopping and watchdog policy.

    HEA permits watchdog handoff and retains L-BFGS-B limit endpoints; small
    comparisons require the stricter handoff and convergence policy. Reject
    unknown experiment names.
    """
    if experiment == "hea":
        return AdamLBFGSBOptions(adam_watchdog_steps=2000,
            adam_handoff_patience=1, adam_watchdog_action="handoff",
            lbfgs_nonconvergence_action="retain_limit")
    if experiment == "small":
        return AdamLBFGSBOptions(adam_watchdog_steps=5000,
            adam_handoff_patience=2, adam_watchdog_action="raise",
            lbfgs_nonconvergence_action="raise")
    raise ValueError(experiment)


def start(setting, index, experiment):
    """Reproduce a deterministic uniform cold start and its PCG64 seed record.

    Args:
        setting: Campaign setting or gradient shape, including its seed keys.
        index: Zero-based start or gradient-sample index.
        experiment: ``hea``, ``small``, or ``gradient`` seed mapping.

    Returns:
        Float64 angles sampled from [0, 2*pi) and seed provenance containing
        the entropy input, spawn key, and generated uint32 state. Historical
        HEA bank indices are preserved across the reduced release matrix.
    """
    if experiment == "small":
        entropy = [2026091101, ("tfim", "xxz").index(setting["model"]),
                   setting["num_system"], setting["beta_index"], index]
        sequence = np.random.SeedSequence(entropy).spawn(2)[("tfda", "hea").index(setting["ansatz_type"])]
    else:
        entropy = 2026091001 if experiment == "gradient" else 2026080701
        if experiment == "gradient":
            key = (setting["seed_namespace"], setting["num_system"], index)
        else:
            key = (setting["seed_bank_index"], index)
            if setting["seed_namespace"] is not None:
                key = (setting["seed_namespace"], *key)
        sequence = np.random.SeedSequence(entropy, spawn_key=key)
    p = make_spec(setting, experiment).num_parameters
    theta = np.random.Generator(np.random.PCG64(sequence)).uniform(0., 2*np.pi, p).astype(np.float64)
    return theta, dict(algorithm="PCG64", entropy=entropy,
        spawn_key=list(sequence.spawn_key), state_uint32=sequence.generate_state(4).tolist())
