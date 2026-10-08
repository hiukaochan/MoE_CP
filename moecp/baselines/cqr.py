"""Conformalized quantile regression (Romano, Patterson & Candes, 2019)."""
import numpy as np

from ..conformal import split_quantile


class CQR:
    """Score E = max(q_lo(x) - y, y - q_hi(x)); interval [q_lo - Q, q_hi + Q]."""

    def __init__(self, qrf, alpha=0.1):
        self.qrf, self.alpha = qrf, alpha
        qs = qrf.quantiles
        self.i_lo = int(np.abs(qs - alpha / 2).argmin())
        self.i_hi = int(np.abs(qs - (1 - alpha / 2)).argmin())

    def _bounds(self, X):
        q = self.qrf.predict(X)
        return q[:, self.i_lo], q[:, self.i_hi]

    def calibrate(self, Xc, yc):
        lo, hi = self._bounds(Xc)
        yc = np.ravel(yc)
        self.q_ = split_quantile(np.maximum(lo - yc, yc - hi), self.alpha)
        return self

    def predict_interval(self, Xt):
        lo, hi = self._bounds(Xt)
        return lo - self.q_, hi + self.q_
