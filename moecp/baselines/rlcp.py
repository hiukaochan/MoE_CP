"""Randomly-localized conformal prediction (Hore & Barber, 2025).

Gaussian-kernel localization around a *randomized* copy of the test point:
  x_tilde ~ N(x_test, I / (2 sigma)),   w_i = exp(-sigma * ||x_i - x_tilde||^2)
over the calibration points and the test point; weighted quantile of |y - mu(x)| with the
test point's weight at +inf.  sigma is set so the kernel's effective local sample size on the
training covariates is `n_eff` (= 100 in the paper).
"""
import numpy as np
from sklearn.metrics import pairwise_distances

from ..conformal import weighted_quantile


def rlcp_bandwidth(X, n_eff=100, lo=0.0, hi=1000.0, tol=1e-3):
    """Binary search for sigma such that the estimated mean local cluster size equals n_eff."""
    n = len(X)
    D2 = pairwise_distances(X, X) ** 2
    while hi - lo > tol:
        sigma = (lo + hi) / 2
        W = np.exp(-sigma * D2)
        np.fill_diagonal(W, 0.0)
        row = W.sum(1)
        n_hat = n * np.mean((row / (n - 1)) ** 2) / (np.sum(W ** 2) / (n * (n - 1)) + 1e-13)
        if n_hat < n_eff:   # kernel too narrow
            hi = sigma
        else:
            lo = sigma
    return (lo + hi) / 2


class RLCP:
    def __init__(self, model, X_train, alpha=0.1, n_eff=100, seed=0):
        self.model, self.alpha = model, alpha
        X_train = np.asarray(X_train).reshape(len(X_train), -1)
        self.mean_, self.std_ = X_train.mean(0), X_train.std(0)
        self.std_[self.std_ == 0] = 1.0
        self.sigma_ = rlcp_bandwidth(self._z(X_train), n_eff)
        self.rng = np.random.default_rng(seed)

    def _z(self, X):
        return (np.asarray(X).reshape(len(X), -1) - self.mean_) / self.std_

    def calibrate(self, Xc, yc):
        R = np.abs(np.ravel(yc) - self.model.predict(np.asarray(Xc).reshape(len(Xc), -1)))
        order = np.argsort(R)
        self.R_, self.Zc_ = R[order], self._z(Xc)[order]
        return self

    def predict_interval(self, Xt):
        Xt2 = np.asarray(Xt).reshape(len(Xt), -1)
        mu = self.model.predict(Xt2)
        Zt = self._z(Xt2)
        Z_tilde = Zt + self.rng.standard_normal(Zt.shape) / np.sqrt(2 * self.sigma_)
        logw_cal = -self.sigma_ * pairwise_distances(Z_tilde, self.Zc_) ** 2      # (m, n)
        logw_test = -self.sigma_ * np.sum((Zt - Z_tilde) ** 2, 1)                 # (m,)
        m = np.maximum(logw_cal.max(1), logw_test)
        w_cal, w_test = np.exp(logw_cal - m[:, None]), np.exp(logw_test - m)
        Q = np.array([weighted_quantile(self.R_, w_cal[j], w_test[j], self.alpha)
                      for j in range(len(Zt))])
        return mu - Q, mu + Q
