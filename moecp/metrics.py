"""Evaluation metrics: marginal coverage, length, local (kNN) coverage, worst-slice coverage."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def covered(lo, hi, y):
    return (np.ravel(y) >= lo) & (np.ravel(y) <= hi)


def coverage(lo, hi, y):
    return covered(lo, hi, y).mean()


def mean_length(lo, hi):
    return np.mean(hi - lo)


def local_coverage_knn(X, cov, k=100):
    """For each point, mean coverage over its k nearest neighbours (paper Fig. 3)."""
    X = np.asarray(X).reshape(len(X), -1)
    _, nbr = NearestNeighbors(n_neighbors=min(k, len(X))).fit(X).kneighbors(X)
    return cov.astype(float)[nbr].mean(1)


def _wsc_slab(Xv, cov, delta, M, rng):
    """Search random directions/slabs on (Xv, cov); return the worst (v, a, b)."""
    n, d = Xv.shape
    V = rng.standard_normal((M, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    min_n = max(int(np.ceil(delta * n)), 1)
    best = (np.inf, None, None, None)
    cov = cov.astype(float)
    for v in V:
        z = Xv @ v
        order = np.argsort(z)
        zs, cs = z[order], np.concatenate([[0.0], np.cumsum(cov[order])])
        # all slabs [i, j] with j - i + 1 >= min_n; vectorized over j for each i
        for i in range(0, n - min_n + 1):
            j = np.arange(i + min_n - 1, n)
            c = (cs[j + 1] - cs[i]) / (j - i + 1)
            jj = np.argmin(c)
            if c[jj] < best[0]:
                best = (c[jj], v, zs[i], zs[j[jj]])
    return best[1], best[2], best[3]


def worst_slice_coverage(X, cov, delta=0.1, M=1000, test_frac=0.8, rng=0):
    """Unbiased worst-slice coverage (Romano et al. 2020; Cauchois et al. 2021).

    The worst slab {x : a <= v'x <= b} holding >= delta of the points is found on one
    split and its coverage is evaluated on the other, so the estimate is not optimistically
    biased by the search.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X).reshape(len(X), -1)
    perm = rng.permutation(len(X))
    n_fit = int(round((1 - test_frac) * len(X)))
    fit, ev = perm[:n_fit], perm[n_fit:]
    v, a, b = _wsc_slab(X[fit], cov[fit], delta, M, rng)
    z = X[ev] @ v
    mask = (z >= a) & (z <= b)
    return cov[ev][mask].mean() if mask.any() else np.nan
