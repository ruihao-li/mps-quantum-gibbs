"""Independent Qiskit/Quimb scoring, preserving production ordering conventions."""
from __future__ import annotations
from common import np
from protocols import make_spec


@np.errstate(all="ignore")
def reconstruct_state(setting, theta):
    """Return cutoff-zero Quimb MPS and ordered logical-system site indices.

    Contiguous states are flipped exactly as in the production trusted scorer.
    Interleaved states retain circuit order and use explicit system positions.
    This public helper is also used by deterministic observable analysis.

    Args:
        setting: Contiguous or interleaved-pair HEA architecture metadata.
        theta: Circuit parameters in the ansatz's parameter order.

    Returns:
        The reconstructed MPS and a tuple of physical sites in logical
        system-spin order. The bond cap uses the exact circuit-depth bound.
    """
    from ansatz import BasicAnsatz, InterleavedPairAnsatz
    from lattice import SquareLattice
    import quimb.tensor as qtn
    from qiskit_quimb import quimb_circuit
    family = setting["ansatz_type"]
    if family not in ("contiguous", "interleaved_pair"):
        raise ValueError("MPS reconstruction supports the two HEA layouts")
    cls = BasicAnsatz if family == "contiguous" else InterleavedPairAnsatz
    ansatz = cls(SquareLattice(setting["nx"], setting["ny"]),
        num_layers=setting["num_layers"], num_ancillas=setting["num_ancillas"])
    cap = (2 if family == "contiguous" else 4) ** setting["num_layers"]
    state = quimb_circuit(ansatz.circuit.assign_parameters(np.asarray(theta)),
        quimb_circuit_class=qtn.CircuitMPS, max_bond=cap, cutoff=0., progbar=False).psi
    if state.max_bond() > cap:
        raise AssertionError("Exact circuit bond bound exceeded")
    if family == "contiguous":
        return state.flip(), tuple(range(setting["num_system"]))
    return state, tuple(ansatz.system_positions)


@np.errstate(all="ignore")
def trusted_mps(setting, theta):
    """Independently score a supported zero-longitudinal-field HEA endpoint.

    Reconstruct the cutoff-zero Qiskit/Quimb MPS, obtain entropy in nats from
    normalized Schmidt weights or the ancilla density matrix, and contract
    normalized X-field and ZZ-bond energy terms. This campaign scorer does
    not include a longitudinal-field contribution.

    Returns:
        Energy, entropy, total free energy E - S/beta, intensive objective
        C_beta = (beta * E - S) / N, normalization, and MPS bond diagnostics.
    """
    state, positions = reconstruct_state(setting, theta)
    n, beta = setting["num_system"], setting["beta"]
    norm = float(complex((state.H & state) ^ ...).real)
    if setting["ansatz_type"] == "contiguous":
        # Flip back before reading precisely the original ancilla-side bond;
        # this also preserves the trusted production Schmidt operation order.
        weights = np.asarray(state.flip().schmidt_values(setting["num_ancillas"]), dtype=np.float64)
    else:
        ancillas = tuple(i for i in range(state.L) if i not in positions)
        rho = np.asarray(state.partial_trace_to_dense_canonical(where=ancillas,
            normalized=True), dtype=np.complex128)
        rho = (rho + rho.conj().T) / 2
        rho /= np.trace(rho).real
        weights = np.linalg.eigvalsh(rho)
    weights = np.clip(weights, 0., None)
    weights /= weights.sum()
    positive = weights > 0.
    entropy = -float(np.sum(weights[positive] * np.log(weights[positive])))
    x = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    z = np.diag([1., -1.]).astype(np.complex128)
    spec = make_spec(setting)
    terms = {(p,): setting["hx"] * x for p in positions}
    terms.update({tuple(sorted((positions[a], positions[b]))): setting["j"] * np.kron(z, z)
                  for a, b in spec.edges})
    energy = float(complex(state.compute_local_expectation_canonical(
        terms, normalized=True, return_all=False)).real)
    result = dict(energy=energy, entropy_nats=entropy, free_energy=energy-entropy/beta,
        C_beta=(beta*energy-entropy)/n, norm=norm, max_bond=int(state.max_bond()),
        bond_dimensions=[int(b) for b in state.bond_sizes()], cutoff=0.)
    if not all(np.isfinite(result[k]) for k in ("energy", "entropy_nats", "free_energy", "C_beta", "norm")):
        raise FloatingPointError("Nonfinite trusted MPS score")
    if abs(norm-1.) > 2e-10:
        raise AssertionError("Trusted MPS norm failed")
    return result


def small_problem(setting):
    """Build the small-system TFDA or HEA ansatz and dense TFIM or XXZ Hamiltonian."""
    from ansatz import BasicAnsatz, TFDAnsatz
    from lattice import SquareLattice
    from spin_hamiltonians import IsingModel, XXZModel
    lattice = SquareLattice(setting["num_system"], 1, periodic=False)
    model = setting["model"]
    hamiltonian = (IsingModel(lattice, j=-1., hz=0., hx=-.5) if model == "tfim"
                   else XXZModel(lattice, jxy=-1., jz=1.5))
    if setting["ansatz_type"] == "tfda":
        ansatz = TFDAnsatz(lattice, model="transverse_ising" if model == "tfim" else "xxz",
                           num_layers=setting["num_layers"])
    else:
        ansatz = BasicAnsatz(lattice, num_layers=setting["num_layers"],
            num_ancillas=setting["num_ancillas"], rotation_blocks="ry",
            entanglement_blocks="cx", entanglement="linear")
    return ansatz, np.asarray(hamiltonian.qubit_op.to_matrix(), dtype=np.complex128)


@np.errstate(all="ignore")
def trusted_small(setting, theta):
    """Score a small-system circuit against an independently diagonalized Gibbs state.

    Trace the first N circuit qubits, Hermitize and normalize the remaining
    density matrix, and compute entropy from eigenvalues above 1e-14.

    Returns:
        Prepared and exact thermodynamics, squared Uhlmann fidelity and
        infidelity, normalization diagnostics, and absolute energy-density
        error. C_beta is (beta * E - S) / N, with entropy measured in nats.
    """
    from qiskit.quantum_info import Statevector, partial_trace, state_fidelity, DensityMatrix
    ansatz, hamiltonian = small_problem(setting)
    n, beta = setting["num_system"], setting["beta"]
    state = Statevector.from_instruction(ansatz.build_circuit().assign_parameters(list(theta)))
    rho = np.asarray(partial_trace(state, qargs=list(range(n))).data)
    trace = np.trace(rho)
    rho = (rho + rho.conj().T) / (2 * trace.real)
    weights = np.linalg.eigvalsh((rho+rho.conj().T)/2).real
    kept = np.where(weights > 1e-14, weights, 0.)
    kept /= kept.sum()
    entropy = -float(np.sum(kept[kept > 1e-14]*np.log(kept[kept > 1e-14])))
    energy = float(np.einsum("ij,ji->", hamiltonian, rho).real)
    energies, vectors = np.linalg.eigh(hamiltonian)
    probabilities = np.exp(-beta*(energies-energies.min()))
    probabilities /= probabilities.sum()
    exact_rho = (vectors*probabilities) @ vectors.conj().T
    # As in the original production scorer, suppress stale BLAS status flags
    # but independently reject nonfinite values and invalid density matrices.
    if not np.isfinite(exact_rho).all() or not np.isfinite(rho).all():
        raise FloatingPointError("Nonfinite independently reconstructed density matrix")
    exact_energy = float(energies @ probabilities)
    exact_entropy = -float(probabilities @ np.log(probabilities))
    fidelity = float(state_fidelity(DensityMatrix(rho), DensityMatrix(exact_rho), validate=True))
    return dict(energy=energy, entropy_nats=entropy, free_energy=energy-entropy/beta,
        C_beta=(beta*energy-entropy)/n, norm=float(trace.real), max_bond=None,
        fidelity=fidelity, infidelity=1-fidelity,
        energy_density_error=abs(energy-exact_energy)/n,
        exact_energy=exact_energy, exact_entropy_nats=exact_entropy,
        exact_C_beta=(beta*exact_energy-exact_entropy)/n,
        density_trace_residual=float(abs(trace-1)), minimum_density_eigenvalue=float(weights.min()))


def score(setting, theta, experiment, internal=None):
    """Validate an independently reconstructed endpoint and optional optimizer agreement.

    Args:
        setting: Campaign architecture and Hamiltonian metadata.
        theta: Parameters of the endpoint to reconstruct.
        experiment: ``small`` for dense scoring, otherwise MPS scoring.
        internal: Optional optimizer metrics for an independent residual check.

    Returns:
        Trusted metrics, including a maximum absolute optimizer residual
        when internal metrics are supplied.

    Raises:
        FloatingPointError: A required thermodynamic field is nonfinite.
        AssertionError: Normalization or optimizer agreement exceeds tolerance.
    """
    result = trusted_small(setting, theta) if experiment == "small" else trusted_mps(setting, theta)
    if not all(np.isfinite(result[k]) for k in ("energy", "entropy_nats", "free_energy", "C_beta", "norm")):
        raise FloatingPointError("Nonfinite trusted endpoint")
    if abs(result["norm"]-1.) > 2e-10:
        raise AssertionError("Trusted endpoint normalization failed")
    if internal is not None:
        residual = max(abs(result[k]-float(getattr(internal, k)))
                       for k in ("energy", "entropy_nats", "free_energy", "norm"))
        result["optimizer_maximum_absolute_residual"] = residual
        if residual > (2e-10 if experiment == "hea" and setting["ansatz_type"] == "contiguous" else 2e-9):
            raise AssertionError(f"Independent reconstruction differs: {residual}")
    return result
