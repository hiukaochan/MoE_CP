"""Split conformal prediction with conditional calibration, SCP+CC (Gibbs, Cherian & Candes, 2025).

Quantile regression of the signed residual S = y - mu(x) on the linear basis Phi(x) = (1, x),
fitted on the calibration points plus the test point with its score imputed.  The cutoff for a
test point x at level q is the imputed score at which the test point's dual variable reaches a
threshold u ~ U(q - 1, q) (randomized, so coverage is exact); it equals Phi(x)' b, where b solves
the tilted quantile regression
    min_b  sum_i rho_q(S_i - Phi(X_i)' b)  -  u Phi(x)' b,
one small LP, solved here through its dual
    max_eta  S' eta   s.t.  Phi' eta = -u Phi(x),   q - 1 <= eta_i <= q.
  lower bound = mu(x) + cutoff at level alpha / 2
  upper bound = mu(x) + cutoff at level 1 - alpha / 2
Coverage then holds under every linear re-weighting of x, not only on average.  The two bounds are
calibrated separately, so they can cross or be infinite.

Configured as `CC` in the PCP reference code, which the paper ran through the authors'
`conditionalconformal` package (exact=True, randomize=True).  That package does not run on
NumPy >= 2.4, hence the direct LP; given the same thresholds the two agree to 1e-14 in most
cutoffs, and where they differ (up to 0.06 on a 14-column basis) the LP is the one that satisfies
the definition above.
"""
import numpy as np
from scipy.optimize import linprog


def _design(X):
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    return np.column_stack([np.ones(len(X)), X])


def cc_cutoff(S, Phi, phi_test, q, u):
    """Score cutoff at level q and dual threshold u.  +-inf if the threshold is never reached."""
    res = linprog(-S, A_eq=Phi.T, b_eq=-u * phi_test, bounds=(q - 1, q), method="highs")
    if res.status != 0:
        return np.inf if u > 0 else -np.inf
    return float(phi_test @ -res.eqlin.marginals)


class SCPCC:
    def __init__(self, model, alpha=0.1, seed=0):
        self.model, self.alpha = model, alpha
        self.rng = np.random.default_rng(seed)

    def calibrate(self, Xc, yc):
        Phi = _design(Xc)
        self.S_ = np.ravel(yc).astype(float) - self.model.predict(Phi[:, 1:])
        # drop linearly dependent basis directions (e.g. a feature constant on the calibration set)
        _, s, Vt = np.linalg.svd(Phi, full_matrices=False)
        self.T_ = Vt.T[:, s > 1e-10]
        self.Phi_ = Phi @ self.T_
        return self

    def predict_interval(self, Xt):
        Z = _design(Xt)
        mu = self.model.predict(Z[:, 1:])
        lo, hi = np.empty(len(Z)), np.empty(len(Z))
        q_lo, q_hi = self.alpha / 2, 1 - self.alpha / 2
        for j, phi in enumerate(Z @ self.T_):
            lo[j] = cc_cutoff(self.S_, self.Phi_, phi, q_lo, self.rng.uniform(q_lo - 1, q_lo))
            hi[j] = cc_cutoff(self.S_, self.Phi_, phi, q_hi, self.rng.uniform(q_hi - 1, q_hi))
        return mu + lo, mu + hi
