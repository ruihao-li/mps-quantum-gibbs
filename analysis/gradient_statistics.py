"""Whole-vector bootstrap for the actual intensive objective C=(βE−S)/N."""
import numpy as np


def summarize(shape, arrays, resamples=5000, seed=2026091002):
    """Summarize paired gradient fluctuations for the intensive objective.

    Args:
        shape: Architecture metadata with system, ancilla, and layer counts.
        arrays: Per-state energies, entropies in nats, and paired gradient
            arrays of shape (samples, parameters).
        resamples: Number of whole-vector bootstrap draws.
        seed: Base seed combined with the architecture-specific spawn key.

    Returns:
        Rows for beta = 0.4, 1, and 3, including unbiased sample variances
        averaged over parameters and percentile 95% bootstrap intervals.
        V_C uses C_beta = (beta * E - S) / N; V_Phi omits the 1/N factor.

    Notes:
        Each bootstrap draw resamples paired energy/entropy gradient vectors
        together, preserving their covariance and within-vector dependence.
        Component quantiles describe variation across parameters, not
        uncertainty in the mean variance.
    """
    ge = np.asarray(arrays['grad_energy'], dtype=float)
    gs = np.asarray(arrays['grad_entropy_nats'], dtype=float)
    energy = np.asarray(arrays['energy'], dtype=float)
    entropy = np.asarray(arrays['entropy_nats'], dtype=float)
    if ge.ndim != 2 or ge.shape != gs.shape or ge.shape[0] < 2:
        raise ValueError("Paired gradient arrays must have shape (samples, parameters)")
    if not all(np.isfinite(a).all() for a in (ge, gs, energy, entropy)):
        raise ValueError("Nonfinite gradient dataset")
    count, p = ge.shape
    n, na, layers = shape['num_system'], shape['num_ancillas'], shape['num_layers']
    if p != (n+na)*layers or energy.shape != (count,) or entropy.shape != (count,):
        raise ValueError("Gradient shape/architecture mismatch")
    centered_e, centered_s = ge-ge.mean(axis=0), gs-gs.mean(axis=0)
    ve, vs = np.var(ge, axis=0, ddof=1).mean(), np.var(gs, axis=0, ddof=1).mean()
    cov = (centered_e*centered_s).sum() / ((count-1)*p)
    # Weighted whole-vector bootstrap: precomputed sample Gram matrices avoid
    # materializing 5000 × samples × parameters copies. Paired E/S samples share
    # each multinomial draw; this is identical to sampling full vectors.
    shape_index = (0 if shape['ansatz_type']=='contiguous' else 6) + (10,12,16,20,24,30).index(n) if n in (10,12,16,20,24,30) else 0
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(shape_index,n)))
    weights = rng.multinomial(count, np.full(count, 1/count), size=resamples).astype(float)
    def variance_bootstrap(samples):
        """Return parameter-averaged unbiased variances for shared bootstrap draws."""
        gram = np.einsum('ip,jp->ij',samples,samples,optimize=True)/samples.shape[1]
        mean_square = np.einsum('bi,i->b',weights,np.diag(gram),optimize=True)
        square_mean = np.einsum('bi,ij,bj->b', weights, gram, weights, optimize=True)/count
        result = (mean_square-square_mean)/(count-1)
        if not np.isfinite(result).all() or np.min(result)<-1e-14:
            raise ValueError('Invalid bootstrap variance')
        return np.maximum(result,0.)
    rows = []
    for beta in (.4, 1., 3.):
        phi = beta*ge-gs
        component = np.var(phi, axis=0, ddof=1)/n**2
        vphi = float(np.var(phi, axis=0, ddof=1).mean())
        np.testing.assert_allclose(vphi, beta**2*ve+vs-2*beta*cov, atol=1e-12, rtol=1e-10)
        bounds = np.quantile(variance_bootstrap(phi), [.025, .975])
        cost = (beta*energy-entropy)/n
        row = dict(ansatz_type=shape['ansatz_type'],
            architecture='Contiguous' if shape['ansatz_type']=='contiguous' else 'Interleaved',
            shape_id=shape['shape_id'], N=n, Na=na, L=layers, P=p, beta=beta, samples=count,
            V_E=float(ve), V_S=float(vs), Cov_ES=float(cov),
            r_ES=float(cov/np.sqrt(ve*vs)) if ve*vs > 0 else 0.,
            V_Phi=vphi, V_C=vphi/n**2,
            V_Phi_ci_low=float(bounds[0]), V_Phi_ci_high=float(bounds[1]),
            V_C_ci_low=float(bounds[0]/n**2), V_C_ci_high=float(bounds[1]/n**2),
            component_V_C_q10=float(np.quantile(component,.1)),
            component_V_C_median=float(np.median(component)),
            component_V_C_q90=float(np.quantile(component,.9)),
            component_V_C_zero_count=int(np.sum(component==0)),
            cost_variance=float(np.var(cost,ddof=1)))
        rows.append(row)
    return rows
