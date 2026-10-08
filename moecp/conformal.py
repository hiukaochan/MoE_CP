"""MoE-weighted conformal prediction (paper Alg. 1 / Alg. 2) and plain split CP.

For a test point x:
  1. pi_tilde = Multinomial(tau, pi(x)) / tau                           Eq. (3)
  2. w_i = exp(-tau * D(pi_tilde, pi(X_i)))  for i in cal and the test   Eq. (4)
  3. normalize over cal + test                                           Eq. (5)
  4. Q = (1-alpha)-quantile of sum_i w_i delta_{S_i} + w_{n+1} delta_{+inf}
  5. C(x) = [mu(x) - Q, mu(x) + Q]            (absolute-residual score)
"""
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- divergences
# Each takes p of shape (K,) or (m, K) and Q of shape (n, K); returns (n,) or (m, n).

def _plogp_q(p, Q):
    p = np.atleast_2d(p)
    logQ = np.log(np.clip(Q, EPS, None))
    return p, logQ


def kl(p, Q):
    """KL(p || q) with 0 log 0 = 0."""
    p, logQ = _plogp_q(p, Q)
    plogp = np.where(p > 0, p * np.log(np.clip(p, EPS, None)), 0.0).sum(1, keepdims=True)
    out = plogp - p @ logQ.T
    return out[0] if out.shape[0] == 1 else out


def cross_entropy(p, Q):
    """H(p, q) = -sum p log q.  Gives the same normalized weights as KL (App. A.4.3)."""
    p, logQ = _plogp_q(p, Q)
    out = -(p @ logQ.T)
    return out[0] if out.shape[0] == 1 else out


def hellinger(p, Q):
    p = np.atleast_2d(p)
    out = np.sqrt(((np.sqrt(p)[:, None, :] - np.sqrt(Q)[None]) ** 2).sum(-1)) / np.sqrt(2)
    return out[0] if out.shape[0] == 1 else out


def euclidean(p, Q):
    p = np.atleast_2d(p)
    out = np.sqrt(((p[:, None, :] - Q[None]) ** 2).sum(-1))
    return out[0] if out.shape[0] == 1 else out


DIVERGENCES = {"kl": kl, "ce": cross_entropy, "hellinger": hellinger, "euclidean": euclidean}


# ---------------------------------------------------------------- building blocks

def randomize(pi_test, tau, rng):
    """pi_tilde = L / tau with L ~ Multinomial(tau, pi_test), row-wise.  Eq. (3)."""
    pi_test = np.atleast_2d(pi_test)
    p = pi_test / pi_test.sum(1, keepdims=True)  # guard against float drift
    L = np.stack([rng.multinomial(int(tau), row) for row in p])
    return L / tau


def log_weights(pi_tilde, pi_ref, tau, divergence="kl"):
    """log w = -tau * D(pi_tilde, pi_ref).  Eq. (4)."""
    return -tau * DIVERGENCES[divergence](pi_tilde, pi_ref)


def weighted_quantile(scores_sorted, w_cal_sorted, w_test, alpha):
    """(1-alpha)-quantile of sum_i w_i delta_{S_i} + w_test delta_{+inf}.

    `scores_sorted` ascending; `w_cal_sorted` the (unnormalized) weights in the same order.
    Normalization includes the test weight (Eq. 5).  Returns +inf if the cal mass never
    reaches 1 - alpha.
    """
    total = w_cal_sorted.sum() + w_test
    cdf = np.cumsum(w_cal_sorted) / total
    idx = np.searchsorted(cdf, 1 - alpha - 1e-12, side="left")
    return scores_sorted[idx] if idx < len(scores_sorted) else np.inf


def split_quantile(scores, alpha):
    """Standard split-CP quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


# ---------------------------------------------------------------- methods

class SplitCP:
    """Split conformal with absolute residual score and uniform weights."""

    def __init__(self, model, alpha=0.1):
        self.model, self.alpha = model, alpha

    def calibrate(self, Xc, yc):
        mu, _ = self.model.predict(Xc)
        self.q_ = split_quantile(np.abs(np.ravel(yc) - mu), self.alpha)
        return self

    def predict_interval(self, Xt):
        mu, _ = self.model.predict(Xt)
        return mu - self.q_, mu + self.q_


class MoECP:
    """MoE-weighted conformal prediction (Alg. 2)."""

    def __init__(self, model, tau=100, divergence="kl", alpha=0.1, seed=0):
        self.model, self.tau, self.divergence = model, tau, divergence
        self.alpha, self.rng = alpha, np.random.default_rng(seed)

    def calibrate(self, Xc, yc):
        mu, pi = self.model.predict(Xc)
        scores = np.abs(np.ravel(yc) - mu)
        order = np.argsort(scores)
        self.scores_, self.pi_cal_ = scores[order], pi[order]
        return self

    def quantiles(self, pi_test, chunk=512):
        """Per-test-point Q_{1-alpha} given the test gating vectors (m, K)."""
        pi_tilde = randomize(pi_test, self.tau, self.rng)
        Q = np.empty(len(pi_test))
        for s in range(0, len(pi_test), chunk):
            pt, pq = pi_tilde[s:s + chunk], pi_test[s:s + chunk]
            lw_cal = np.atleast_2d(log_weights(pt, self.pi_cal_, self.tau, self.divergence))  # (c, n)
            lw_test = np.diagonal(np.atleast_2d(
                log_weights(pt, pq, self.tau, self.divergence)))                              # (c,)
            m = np.maximum(lw_cal.max(1), lw_test)  # log-space stabilization
            w_cal, w_test = np.exp(lw_cal - m[:, None]), np.exp(lw_test - m)
            for j in range(len(pt)):
                Q[s + j] = weighted_quantile(self.scores_, w_cal[j], w_test[j], self.alpha)
        return Q

    def predict_interval(self, Xt):
        mu, pi = self.model.predict(Xt)
        Q = self.quantiles(pi)
        return mu - Q, mu + Q
