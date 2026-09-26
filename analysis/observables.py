"""Normalized TFIM moments with explicit logical-to-MPS site mapping.

The transfer contractions follow the validated endpoint diagnostics. H and H²
are independently contracted as MPO networks, without compression. No dense
system density matrix is constructed, and entropy is not rediagonalized here.
"""
import math
import numpy as np

I2 = np.eye(2)
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])


def real(value):
    """Return a finite expectation's real part, rejecting significant imaginary parts."""
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise FloatingPointError("Nonfinite expectation")
    if abs(value.imag) > 2e-10 * max(1., abs(value.real)):
        raise FloatingPointError("Complex expectation of a Hermitian operator")
    return value.real


def tensors_lsr(state):
    """Return qubit MPS arrays ordered as left bond, physical site, right bond."""
    arrays = []
    for i in range(state.L):
        inds = ([state.bond(i-1, i)] if i else []) + [state.site_ind(i)]
        inds += [state.bond(i, i+1)] if i+1 < state.L else []
        shape = (state.bond_size(i-1, i) if i else 1, 2,
                 state.bond_size(i, i+1) if i+1 < state.L else 1)
        arrays.append(np.asarray(state[i].transpose(*inds).data).reshape(shape))
    return arrays


def advance(left, a, op=I2):
    """Contract a left environment through one MPS site and local operator."""
    out = np.zeros((a.shape[2], a.shape[2]), dtype=np.complex128)
    for bra, ket in zip(*np.nonzero(op)):
        out += op[bra, ket] * (a[:, bra, :].conj().T @ left @ a[:, ket, :])
    return out


def retreat(right, a):
    """Contract a right identity environment through one qubit MPS site."""
    return sum(a[:, s, :].conj() @ right @ a[:, s, :].T for s in range(2))


def local_moments(state, positions):
    """Measure normalized X, Z, and ZZ moments in logical-system order.

    Args:
        state: Qubit MPS for the full system-ancilla pure state.
        positions: Distinct MPS site indices ordered by logical system spin.

    Returns:
        The full-state squared norm, X and Z expectation vectors, and the
        raw ZZ correlation matrix, including unit diagonal entries.

    Raises:
        ValueError: The logical-to-MPS site map is invalid.
        AssertionError: Left and right normalization contractions disagree.
    """
    if len(set(positions)) != len(positions) or any(i < 0 or i >= state.L for i in positions):
        raise ValueError("Invalid logical-system site map")
    arrays = tensors_lsr(state)
    left = [np.ones((1, 1), dtype=np.complex128)]
    for a in arrays:
        left.append(advance(left[-1], a))
    right = [None] * (len(arrays)+1)
    right[-1] = np.ones((1, 1), dtype=np.complex128)
    for i in reversed(range(len(arrays))):
        right[i] = retreat(right[i+1], arrays[i])
    norm = real(left[-1][0, 0])
    if norm <= 0 or abs(norm-real(right[0][0, 0])) > 3e-10 * norm:
        raise AssertionError("Left/right normalization disagrees")
    one = [np.array([real(np.sum(advance(left[i], arrays[i], op)*right[i+1])) / norm
                     for i in positions]) for op in (X, Z)]
    index = {site: i for i, site in enumerate(positions)}
    zz = np.eye(len(positions))
    # Sort physical positions; preserve logical ordering in the returned matrix.
    for i in sorted(positions):
        inserted = advance(left[i], arrays[i], Z)
        for j in range(i+1, len(arrays)):
            if j in index:
                zz[index[i], index[j]] = zz[index[j], index[i]] = real(np.sum(
                    advance(inserted, arrays[j], Z)*right[j+1])) / norm
            inserted = advance(inserted, arrays[j])
    return norm, one[0], one[1], zz


def edges(nx, ny):
    """Return open rectangular-lattice bonds using row-major site indices."""
    return ([(y*nx+x, y*nx+x+1) for y in range(ny) for x in range(nx-1)] +
            [(y*nx+x, (y+1)*nx+x) for y in range(ny-1) for x in range(nx)])


def stable_variance(second, mean):
    """Subtract the squared mean, clipping only roundoff-scale negative variance."""
    raw = second - mean*mean
    if raw < -2e-10 * max(1., abs(second), mean*mean):
        raise FloatingPointError("Negative variance beyond roundoff")
    return max(0., raw)


def measure(state, positions, setting):
    """Measure normalized TFIM moments and prepared-state fluctuation proxies.

    Args:
        state: Full system-ancilla MPS, contracted without MPO compression.
        positions: MPS sites in logical system-spin order.
        setting: Lattice dimensions, system size, beta, and signed couplings.

    Returns:
        Moments and resource diagnostics. ``correlations`` contains raw ZZ
        moments; ``connected`` subtracts the product of the prepared state's
        Z means. ``chi`` is beta times the summed connected covariance per
        spin, and ``cv`` is beta squared times the energy variance per spin.

    Notes:
        These fluctuation proxies are evaluated in the prepared state; no
        beta or field derivative of an optimized state is taken. Energy is
        cross-checked against local moments, and H squared is independently
        contracted as two uncompressed MPO insertions.
    """
    from quimb.experimental.operatorbuilder import HilbertSpace, SparseOperatorBuilder
    import quimb.tensor as qtn
    n, beta = setting['num_system'], setting['beta']
    links = edges(setting['nx'], setting['ny'])
    norm, x, z, zz = local_moments(state, positions)
    connected = zz - np.outer(z, z)
    if np.linalg.eigvalsh(connected)[0] < -2e-9:
        raise AssertionError("Longitudinal covariance is not positive semidefinite")
    if max(np.max(np.abs(x)), np.max(np.abs(z)), np.max(np.abs(zz))) > 1+2e-9:
        raise AssertionError("Unphysical Pauli expectation")
    builder = SparseOperatorBuilder(hilbert_space=HilbertSpace(sites=state.L))
    for a, b in links:
        builder += setting['j'], ('z', positions[a]), ('z', positions[b])
    for i in positions:
        builder += setting['hx'], ('x', i)
        if setting.get('hz', 0.):
            builder += setting['hz'], ('z', i)
    h = builder.build_mpo()
    energy = real(qtn.expec_TN_1D(state.H, h, state, compress=False)) / norm
    second = real(qtn.expec_TN_1D(state.H, h, h, state, compress=False)) / norm
    ezz = setting['j'] * sum(zz[a, b] for a, b in links)
    ex = setting['hx'] * x.sum()
    ez = setting.get('hz', 0.) * z.sum()
    np.testing.assert_allclose(energy, ezz+ex+ez, atol=2e-8, rtol=2e-10)
    variance = stable_variance(second, energy)
    return dict(norm=norm, energy=energy, energy_squared=second,
                energy_variance=variance, chi=beta*float(connected.sum())/n,
                cv=beta**2*variance/n, ezz_density=float(ezz)/n,
                ex_density=float(ex)/n, x_expectations=x.tolist(),
                z_expectations=z.tolist(), correlations=zz.tolist(),
                connected=connected.tolist(), max_bond=int(state.max_bond()))


def distance_average(connected, nx, ny):
    """Average supplied correlations over unordered pairs at equal distance.

    Use chain separation or rectangular-lattice Euclidean radial shells,
    excluding self-pairs. The caller supplies connected correlations or
    symmetry-equivalent reference moments; no mean subtraction is done here.
    """
    matrix = np.asarray(connected)
    if matrix.shape != (nx*ny, nx*ny):
        raise ValueError("Wrong correlation matrix shape")
    groups = {}
    for i in range(nx*ny):
        for j in range(i+1, nx*ny):
            r2 = (i % nx-j % nx)**2 + (i // nx-j // nx)**2
            groups.setdefault(r2, []).append(matrix[i, j])
    return [dict(distance_squared=r2, distance=math.sqrt(r2), pair_count=len(values),
                 connected_correlation=float(np.mean(values)))
            for r2, values in sorted(groups.items())]
