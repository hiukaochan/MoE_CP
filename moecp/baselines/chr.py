"""Conformal histogram regression (Sesia & Romano, 2021).

1. Turn the quantile-forest quantiles at each x into a histogram over a fixed y-grid.
2. Build a nested family of intervals: at level a, the shortest contiguous run of bins with
   mass >= 1 - a, nested across the a-grid (shrinking inside / growing around the interval
   at the target level), with optional randomized removal of one endpoint bin.
3. Score of a calibration point = 1 - (largest a whose interval still contains y); the
   conformal quantile of the scores gives the calibrated level a_hat used at test time.

Vectorized over points (the reference implementation loops over points in Python).
"""
import numpy as np

from ..conformal import split_quantile

EPS = 1e-12


def _window_sums(C, starts, lengths):
    """C: (n, B+1) cumulative masses; starts/lengths broadcastable int arrays -> masses."""
    rows = np.arange(C.shape[0]).reshape(-1, *([1] * (np.ndim(starts) - 1)))
    return C[rows, starts + lengths] - C[rows, starts]


def _shortest_unconstrained(C, m):
    """Per row: shortest window with mass >= m (binary search on length); returns (start, length)."""
    n, B = C.shape[0], C.shape[1] - 1
    lo, hi = np.ones(n, int), np.full(n, B)
    s_all = np.arange(B)[None, :]
    while np.any(lo < hi):
        mid = (lo + hi) // 2
        valid = s_all + mid[:, None] <= B
        e = np.where(valid, s_all + mid[:, None], B)
        sums = np.where(valid, np.take_along_axis(C, e, 1) - C[:, :B], -np.inf)
        ok = sums.max(1) >= m - EPS
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid + 1)
    valid = s_all + lo[:, None] <= B
    e = np.where(valid, s_all + lo[:, None], B)
    sums = np.where(valid, np.take_along_axis(C, e, 1) - C[:, :B], -np.inf)
    return sums.argmax(1), lo


def _shrink(C, a, L0, m):
    """Shortest window inside [a, a+L0) with mass >= m (current window always feasible)."""
    start, L = a.copy(), L0.copy()
    active = np.ones(len(a), bool)
    d = 0
    while active.any():
        d += 1
        idx = np.flatnonzero(active)
        Lc = L0[idx] - d
        o = np.arange(d + 1)[None, :]
        s = a[idx, None] + o
        valid = (Lc[:, None] >= 1) & (o <= d)
        sums = np.where(valid, _window_sums(C[idx], s, np.maximum(Lc, 1)[:, None]), -np.inf)
        ok = sums.max(1) >= m - EPS
        good = idx[ok]
        start[good] = a[good] + sums[ok].argmax(1)
        L[good] = Lc[ok]
        active[idx[~ok]] = False
    return start, L


def _expand(C, a, L0, m):
    """Shortest window containing [a, a+L0) with mass >= m."""
    B = C.shape[1] - 1
    start, L = a.copy(), L0.copy()
    done = _window_sums(C, a, L0) >= m - EPS
    d = 0
    while not done.all():
        d += 1
        idx = np.flatnonzero(~done)
        Lc = L0[idx] + d
        s = a[idx, None] - np.arange(d + 1)[None, :]
        valid = (s >= 0) & (s + Lc[:, None] <= B)
        s_safe = np.clip(s, 0, B - Lc[:, None].clip(max=B))
        sums = np.where(valid, _window_sums(C[idx], s_safe, Lc[:, None]), -np.inf)
        ok = sums.max(1) >= m - EPS
        good = idx[ok]
        start[good] = s[ok, sums[ok].argmax(1)]
        L[good] = Lc[ok]
        done[good] = True
    return start, L


class CHR:
    def __init__(self, qrf, alpha=0.1, ymin=-1.0, ymax=1.0, n_bins=1000, delta_alpha=0.001,
                 randomize=True, seed=0):
        self.qrf, self.alpha, self.randomize = qrf, alpha, randomize
        self.edges = np.linspace(ymin, ymax, n_bins + 1)
        grid = np.round(np.arange(delta_alpha, 1.0, delta_alpha), 6)
        self.grid = np.unique(np.concatenate([grid, [alpha]]))
        self.k_star = int(np.flatnonzero(np.isclose(self.grid, alpha))[0])
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ histogram
    def histogram(self, X):
        q = self.qrf.predict(X)
        n = len(q)
        lo, hi = self.edges[0], self.edges[-1]
        levels = np.concatenate([[0.0], self.qrf.quantiles, [1.0]])
        knots = np.column_stack([np.full(n, lo), np.clip(q, lo, hi), np.full(n, hi)])
        knots = knots + np.sort(self.rng.uniform(0, 1e-5, knots.shape), 1)  # strictly increasing
        cdf = np.vstack([np.interp(self.edges, k, levels) for k in knots])
        p = np.diff(cdf, axis=1) + 1e-6
        return p / p.sum(1, keepdims=True)                                   # (n, n_bins)

    # ------------------------------------------------------------------ nested intervals
    def interval_sequence(self, p):
        """Randomized nested intervals for every level in self.grid -> (start, length), each (G, n)."""
        n, B = p.shape
        C = np.concatenate([np.zeros((n, 1)), np.cumsum(p, 1)], 1)
        u = self.rng.uniform(size=n) if self.randomize else None
        G = len(self.grid)
        S, L = np.zeros((G, n), int), np.zeros((G, n), int)       # deterministic
        Sr, Lr = np.zeros((G, n), int), np.zeros((G, n), int)     # randomized

        def randomize(s, l, a):
            if u is None:
                return s, l
            s, l = s.copy(), l.copy()
            mass = _window_sums(C, s, l)
            excess = mass - (1 - a)
            w_left = p[np.arange(n), s]
            w_right = p[np.arange(n), s + l - 1]
            can = l > 2
            left = w_left < w_right
            prob = excess / (np.where(left, w_left, w_right) + 1e-5)
            drop = can & (u <= prob)
            s = np.where(drop & left, s + 1, s)
            l = np.where(drop, l - 1, l)
            return s, l

        k = self.k_star
        S[k], L[k] = _shortest_unconstrained(C, 1 - self.grid[k])
        Sr[k], Lr[k] = randomize(S[k], L[k], self.grid[k])
        for k in range(self.k_star + 1, G):                      # larger alpha: shrink
            S[k], L[k] = _shrink(C, S[k - 1], L[k - 1], 1 - self.grid[k])
            Sr[k], Lr[k] = randomize(S[k], L[k], self.grid[k])
        for k in range(self.k_star - 1, -1, -1):                 # smaller alpha: grow
            S[k], L[k] = _expand(C, Sr[k + 1], Lr[k + 1], 1 - self.grid[k])
            Sr[k], Lr[k] = randomize(S[k], L[k], self.grid[k])
        return Sr, Lr

    # ------------------------------------------------------------------ conformal API
    def calibrate(self, Xc, yc):
        Sr, Lr = self.interval_sequence(self.histogram(Xc))
        lo, hi = self.edges[Sr], self.edges[Sr + Lr]
        y = np.ravel(yc)[None, :]
        inside = (y >= lo) & (y <= hi)                                   # (G, n)
        alpha_max = np.where(inside, self.grid[:, None], 0.0).max(0)
        scores = 1.0 - alpha_max
        self.alpha_hat_ = 1.0 - split_quantile(scores, self.alpha)
        return self

    def predict_interval(self, Xt):
        Sr, Lr = self.interval_sequence(self.histogram(Xt))
        ok = np.flatnonzero(self.grid <= self.alpha_hat_ + 1e-12)
        if len(ok) == 0:                                                 # full y-range
            n = Sr.shape[1]
            return np.full(n, self.edges[0]), np.full(n, self.edges[-1])
        j = ok.max()
        return self.edges[Sr[j]], self.edges[Sr[j] + Lr[j]]
