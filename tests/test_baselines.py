"""Coverage / structure checks for CQR, CHR, SCP+CC, RLCP, PCP. Run: python tests/test_baselines.py"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moecp.baselines.cc import SCPCC, cc_cutoff  # noqa: E402
from moecp.baselines.chr import CHR  # noqa: E402
from moecp.baselines.cqr import CQR  # noqa: E402
from moecp.baselines.pcp import PCP  # noqa: E402
from moecp.baselines.qrf import QRF  # noqa: E402
from moecp.baselines.rlcp import RLCP  # noqa: E402
from moecp.conformal import split_quantile  # noqa: E402

ALPHA = 0.1


def _data(n, rng):
    X = rng.uniform(-2, 2, size=(n, 1))
    y = np.sin(2 * X[:, 0]) + (0.2 + np.abs(X[:, 0])) * rng.standard_normal(n)
    return X, y


def _coverage_sim(make_method, trials, n=300, n_test=100):
    covs = []
    for t in range(trials):
        rng = np.random.default_rng(t)
        (Xtr, ytr), (Xc, yc), (Xt, yt) = _data(n, rng), _data(n, rng), _data(n_test, rng)
        lo, hi = make_method(Xtr, ytr, t).calibrate(Xc, yc).predict_interval(Xt)
        covs.append(np.mean((yt >= lo) & (yt <= hi)))
    covs = np.array(covs)
    return covs.mean(), covs.std(ddof=1) / np.sqrt(trials)


def _check(name, make_method, trials):
    cov, se = _coverage_sim(make_method, trials)
    print(f"  {name}: coverage = {cov:.4f} ± {se:.4f}")
    assert cov >= 1 - ALPHA - 3 * se, name


def test_cqr_coverage():
    _check("CQR", lambda X, y, s: CQR(QRF(seed=s).fit(X, y), ALPHA), 150)


def test_rlcp_coverage():
    _check("RLCP", lambda X, y, s: RLCP(RandomForestRegressor(random_state=s).fit(X, y), X,
                                         ALPHA, seed=s), 150)


def test_scpcc_intercept_only_is_split():
    """With the basis Phi = 1 and no randomization the cutoff is the split-conformal quantile."""
    rng = np.random.default_rng(0)
    for n in [9, 50, 199]:
        s = rng.standard_normal(n)
        for alpha in [0.05, 0.1, 0.3]:
            q = cc_cutoff(s, np.ones((n, 1)), np.ones(1), 1 - alpha, 1 - alpha - 1e-9)
            assert np.isclose(q, split_quantile(s, alpha))   # also true when both are +inf


def test_scpcc_coverage():
    _check("SCP+CC", lambda X, y, s: SCPCC(RandomForestRegressor(random_state=s).fit(X, y),
                                           ALPHA, seed=s), 50)


def _chr(X, y, s, **kw):
    r = np.ptp(y)
    return CHR(QRF(seed=s).fit(X, y), ALPHA, y.min() - 0.1 * r, y.max() + 0.1 * r,
               seed=s, **kw)


def test_chr_nested():
    rng = np.random.default_rng(0)
    X, y = _data(300, rng)
    m = _chr(X, y, 0, randomize=False)
    S, L = m.interval_sequence(m.histogram(X[:100]))
    assert np.all(S[1:] >= S[:-1]) and np.all(S[1:] + L[1:] <= S[:-1] + L[:-1])


def test_chr_coverage():
    _check("CHR", lambda X, y, s: _chr(X, y, s, delta_alpha=0.01), 100)


def test_pcp_smoke():
    rng = np.random.default_rng(0)
    (Xtr, ytr), (Xc, yc), (Xt, yt) = _data(300, rng), _data(300, rng), _data(100, rng)
    p = PCP(RandomForestRegressor(random_state=0).fit(Xtr, ytr), ALPHA, seed=0).train(Xtr, ytr)
    lo, hi = p.calibrate(Xc, yc).predict_interval(Xt)
    cov = np.mean((yt >= lo) & (yt <= hi))
    print(f"  PCP: n_c={p.n_c_}, m={p.m_}, coverage={cov:.3f}, inf={np.mean(~np.isfinite(hi)):.3f}")
    assert 0.8 <= cov <= 1.0 and np.all(hi >= lo)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
