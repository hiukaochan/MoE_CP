"""Sanity checks for the conformal machinery. Run: python tests/test_conformal.py (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moecp.conformal import (MoECP, SplitCP, cross_entropy, kl, split_quantile,  # noqa: E402
                             weighted_quantile)


class OracleGate:
    """Known 'model': mu(x) = 0, pi(x) given. Lets us test calibration in isolation."""

    def __init__(self, pi):
        self.pi = pi

    def predict(self, idx):
        idx = np.asarray(idx).ravel().astype(int)
        return np.zeros(len(idx)), self.pi[idx]


def test_weighted_quantile_uniform_matches_split():
    rng = np.random.default_rng(0)
    for n in [9, 50, 199]:
        s = np.sort(rng.standard_normal(n))
        for alpha in [0.05, 0.1, 0.3]:
            assert weighted_quantile(s, np.ones(n), 1.0, alpha) == split_quantile(s, alpha)


def test_weighted_quantile_infinite():
    s = np.arange(5.0)
    # test weight carries > alpha of the mass -> unreachable -> +inf
    assert np.isinf(weighted_quantile(s, np.ones(5), 10.0, 0.1))


def test_kl_ce_same_normalized_weights():
    rng = np.random.default_rng(1)
    Q = rng.dirichlet(np.ones(4), size=30)
    p = rng.dirichlet(np.ones(4))
    for tau in [5, 100]:
        a, b = np.exp(-tau * kl(p, Q)), np.exp(-tau * cross_entropy(p, Q))
        assert np.allclose(a / a.sum(), b / b.sum())


def test_small_tau_close_to_split():
    rng = np.random.default_rng(2)
    n = 400
    pi = rng.dirichlet(np.ones(3), size=n + 1)
    y = rng.standard_normal(n + 1)
    model = OracleGate(pi)
    q_split = SplitCP(model, 0.1).calibrate(np.arange(n), y[:n]).q_
    m = MoECP(model, tau=1, alpha=0.1, seed=0).calibrate(np.arange(n), y[:n])
    q_moe = m.quantiles(pi[n:])[0]
    assert abs(q_moe - q_split) < 0.25 * q_split


def test_theorem1_marginal_coverage():
    """Exchangeable data, latent domains with different noise, fixed gate -> coverage >= 1-alpha."""
    rng = np.random.default_rng(3)
    trials, n, K, alpha, tau = 2000, 200, 3, 0.1, 50
    sig = np.array([0.3, 1.0, 3.0])
    hits = 0
    for _ in range(trials):
        dom = rng.integers(K, size=n + 1)
        pi = rng.dirichlet(np.ones(K) * 0.5, size=n + 1)
        pi = 0.7 * np.eye(K)[dom] + 0.3 * pi                  # informative but noisy gate
        y = rng.standard_normal(n + 1) * sig[dom]
        m = MoECP(OracleGate(pi), tau=tau, alpha=alpha, seed=int(rng.integers(1 << 31)))
        m.calibrate(np.arange(n), y[:n])
        hits += abs(y[n]) <= m.quantiles(pi[n:])[0]
    cov = hits / trials
    se = np.sqrt(alpha * (1 - alpha) / trials)
    print(f"  theorem-1 coverage = {cov:.4f} (target >= {1 - alpha - 3 * se:.4f})")
    assert cov >= 1 - alpha - 3 * se


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
