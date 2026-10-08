"""Posterior conformal prediction (Zhang & Candes, 2024), regression version.

Training stage (on cross-validated residuals R of the base model on the training split):
  * grid xi_1..xi_9 = 10th..90th percentiles of R; for each grid point a linear "odds"
    classifier estimates P(R <= xi_t | x) = s / (1 + s), s = max(x'b, 0),
    b = (n0 / n1) (X0'X0 + lambda n0 I)^-1 sum(X1), lambda chosen per t by CV (AUC);
  * a mixture model on the (n, 9) matrix of these estimates (k-means++ init, then
    alternating mirror descent on memberships / least squares on centres) gives membership
    probabilities; the number of components and the number m of multinomial draws are tuned.
Calibration stage (per test point x, scanning hypothesised residual bins from the top):
  * cross-fit the classifiers on the calibration set with x added under the hypothesised
    label, refit the mixture on the (n+1, 9) matrix, draw m component labels from x's
    memberships, weight each point by prod_k prob_i[k]^(count_k) (the same multinomial
    likelihood form as MoE-CP), and take the weighted quantile with x placed at the bin's
    upper edge; stop at the first bin whose quantile lies inside it.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

from ..conformal import weighted_quantile

LAMBDAS = [10.0 ** i for i in range(8, -8, -1)]


# ---------------------------------------------------------------- mixture model

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _clip(p, eps=1e-8):
    return np.maximum(p, eps)


def _kmeanspp(Y, k, rng, n_init=20):
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        C = [Y[rng.integers(len(Y))]]
        d2 = ((Y - C[0]) ** 2).sum(1)
        for _ in range(1, k):
            C.append(Y[np.argmax(rng.random(len(Y)) * d2)])
            d2 = np.minimum(d2, ((Y - C[-1]) ** 2).sum(1))
        if d2.sum() < best_inertia:
            best, best_inertia = np.array(C), d2.sum()
    d2_all = ((Y[:, None, :] - best[None]) ** 2).sum(-1)
    return best, _softmax(-0.5 * d2_all)


def _mirror_descent(Y, V, w, lr=0.1, steps=20):
    for t in range(steps):
        grad = (w @ V - Y) @ V.T
        w = _softmax(np.log(_clip(w)) - lr * grad / np.sqrt(lr * t + 1))
    return w


def mixture(P, k, seed=0, max_iter=20, tol=1e-3):
    """Fit memberships w (n, k) and centres V (k, d) so that w @ V reconstructs standardized P."""
    Y = np.array(P, dtype=float)
    Y[:, np.isnan(Y).any(0)] = 1.0
    sd = Y.std(0)
    Y = (Y - Y.mean(0)) / np.where(sd > 0, sd, 1.0)
    V, w = _kmeanspp(Y, k, np.random.default_rng(seed))
    r2_old = r2_score(Y, w @ V)
    r2 = r2_old
    for _ in range(max_iter):
        w = _mirror_descent(Y, V, w)
        V = np.linalg.pinv(w.T @ w) @ w.T @ Y
        r2 = r2_score(Y, w @ V)
        if abs(r2 - r2_old) <= tol:
            break
        r2_old = r2
    return w, V, r2


# ---------------------------------------------------------------- odds classifier

def _coef(X0tX0, h, n0, n1, lam):
    """b = (n0/n1) (X0'X0 + lam n0 I)^-1 h."""
    A = X0tX0 + lam * n0 * np.eye(len(h))
    return (n0 / max(n1, 1)) * np.linalg.solve(A, h)


def _prob(X, b):
    s = np.maximum(X @ b, 0.0)
    return s / (s + 1.0)


class PCP:
    def __init__(self, model, alpha=0.1, folds=20, grid=9, l=10, u=90, m_min=5, seed=0):
        self.model, self.alpha, self.folds = model, alpha, folds
        self.levels = np.linspace(l, u, grid)
        self.m_min, self.seed = m_min, seed

    # -------------------------------------------------------------- helpers
    def _design(self, X):
        Z = (np.asarray(X).reshape(len(X), -1) - self.mean_) / self.std_
        return np.column_stack([np.ones(len(Z)), Z])

    # -------------------------------------------------------------- training stage
    def train(self, X_train, y_train, cv_splits=10):
        X_train = np.asarray(X_train).reshape(len(X_train), -1)
        y_train = np.ravel(y_train)
        # cross-validated residuals of the base model on the training split
        R = np.empty(len(y_train))
        for tr, va in KFold(cv_splits, shuffle=True, random_state=self.seed).split(X_train):
            m = clone(self.model).fit(X_train[tr], y_train[tr])
            R[va] = np.abs(y_train[va] - m.predict(X_train[va]))

        self.mean_, self.std_ = X_train.mean(0), X_train.std(0)
        self.std_[self.std_ == 0] = 1.0
        X = self._design(X_train)
        self.xi_ = np.percentile(R, self.levels)
        Y = (R[:, None] <= self.xi_).astype(float)
        P = self._select_lambda(X, Y)
        self.n_c_ = self._n_components(P)
        prob, _, self.r2_ = mixture(P, self.n_c_, self.seed)
        self.m_ = max(int(self._n_draws(prob)), self.m_min)
        return self

    def _select_lambda(self, X, Y):
        T = Y.shape[1]
        kf = list(KFold(self.folds).split(X))
        loss = np.zeros((len(LAMBDAS), T))
        preds = np.zeros((len(LAMBDAS),) + Y.shape)
        for f, (tr, te) in enumerate(kf):
            for t in range(T):
                y0 = Y[tr, t] == 0
                X0tX0, h = X[tr][y0].T @ X[tr][y0], X[tr][~y0].sum(0)
                for li, lam in enumerate(LAMBDAS):
                    p = _prob(X[te], _coef(X0tX0, h, y0.sum(), (~y0).sum(), lam))
                    preds[li, te, t] = p
                    yt = Y[te, t]
                    if yt.min() == yt.max():
                        loss[li, t] -= np.mean((p >= 0.5) * yt) / len(kf)
                    else:
                        loss[li, t] -= roc_auc_score(yt, p) / len(kf)
        best = loss.argmin(0)
        self.lam_ = np.array(LAMBDAS)[best]
        return np.stack([preds[best[t], :, t] for t in range(T)], 1)

    def _n_components(self, P):
        k, r0, r = 1, -1.0, 0.0
        while abs(r - r0) > 0.05:
            k += 1
            r0 = r
            _, _, r = mixture(P, k, self.seed)
        return k - 1 if abs(r - r0) <= 0.025 else k

    def _n_draws(self, prob):
        """Binary search for m: mean effective sample size <= 100 and mean self-weight >= 1/30."""
        n = min(len(prob), 1000)
        logp = np.log(_clip(prob[:n]))
        lo, hi = 4.0, 501.0
        m = (lo + hi) / 2
        rng = np.random.default_rng(self.seed)
        while hi - lo > 2:
            counts = np.stack([rng.multinomial(int(m), prob[j] / prob[j].sum()) for j in range(n)])
            lw = counts @ logp.T                                  # (n, n): row j = weights for test j
            W = np.exp(lw - lw.max(1, keepdims=True))
            W /= W.sum(1, keepdims=True)
            n_hat = np.mean(1.0 / np.sum(W ** 2, 1))
            self_w = np.mean(np.diag(W))
            if n_hat <= 100 and self_w >= 1 / 30:
                hi = m
            else:
                lo = m
            m = (lo + hi) / 2
        return m

    # -------------------------------------------------------------- calibration stage
    def calibrate(self, Xc, yc):
        Xc2 = np.asarray(Xc).reshape(len(Xc), -1)
        self.R_ = np.abs(np.ravel(yc) - self.model.predict(Xc2))
        X = self._design(Xc2)
        Y = (self.R_[:, None] <= self.xi_)
        self.Xc_ = X
        self.fold_idx_ = [te for _, te in KFold(self.folds).split(X)]
        T, d = len(self.xi_), X.shape[1]
        # per fold f < last: models trained on the other folds (incl. the last fold, which will
        # also contain the test point); per fold == last: trained on folds f < last only.
        self.stats_ = []
        for f, te in enumerate(self.fold_idx_):
            tr = np.setdiff1d(np.arange(len(X)), te)
            st = []
            for t in range(T):
                y0 = ~Y[tr, t]
                st.append(dict(G=X[tr][y0].T @ X[tr][y0], h=X[tr][~y0].sum(0),
                               n0=int(y0.sum()), n1=int((~y0).sum()), lam=self.lam_[t]))
            self.stats_.append(st)
        return self

    def _estimates(self, x):
        """Return (E0, E1): (n+1, T) estimate matrices with x labelled 0 / 1 in each column."""
        n, T = len(self.Xc_), len(self.xi_)
        E0, E1 = np.zeros((n + 1, T)), np.zeros((n + 1, T))
        last = len(self.fold_idx_) - 1
        for f, te in enumerate(self.fold_idx_):
            Xf = self.Xc_[te]
            for t in range(T):
                s = self.stats_[f][t]
                if f == last:   # model does not see x; also predicts x itself
                    b = _coef(s["G"], s["h"], s["n0"], s["n1"], s["lam"])
                    E0[te, t] = E1[te, t] = _prob(Xf, b)
                    E0[n, t] = E1[n, t] = _prob(x[None], b)[0]
                else:           # model trained with x added under label 0 or 1
                    b0 = _coef(s["G"] + np.outer(x, x), s["h"], s["n0"] + 1, s["n1"], s["lam"])
                    b1 = _coef(s["G"], s["h"] + x, s["n0"], s["n1"] + 1, s["lam"])
                    E0[te, t], E1[te, t] = _prob(Xf, b0), _prob(Xf, b1)
        return E0, E1

    def quantile(self, x, rng):
        n, T = len(self.Xc_), len(self.xi_)
        edges = np.concatenate([[0.0], self.xi_, [np.inf]])
        E0, E1 = self._estimates(x)
        draw_seed = int(rng.integers(1 << 31))
        for t in range(T, -1, -1):                    # hypothesis: R in (edges[t], edges[t+1]]
            E = np.where(np.arange(T)[None, :] >= t, E1, E0)
            prob, _, _ = mixture(E, self.n_c_, self.seed, max_iter=10, tol=0.005)
            p_test = prob[-1] / prob[-1].sum()
            counts = np.random.default_rng(draw_seed).multinomial(self.m_, p_test)
            lw = np.log(_clip(prob)) @ counts
            w = np.exp(lw - lw.max())
            data = np.append(self.R_, edges[t + 1])
            order = np.argsort(data, kind="stable")
            r = weighted_quantile(data[order], w[order], 0.0, self.alpha)
            if r >= edges[t]:
                return r
        return r

    def predict_interval(self, Xt):
        Xt2 = np.asarray(Xt).reshape(len(Xt), -1)
        mu = self.model.predict(Xt2)
        Z = self._design(Xt2)
        rng = np.random.default_rng(self.seed)
        Q = np.array([self.quantile(z, rng) for z in Z])
        return mu - Q, mu + Q
