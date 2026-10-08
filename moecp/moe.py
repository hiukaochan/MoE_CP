"""Mixture-of-Experts regressor (paper Sec. 2.2, App. A.4.1).

K experts mu_k(x) and a gating network pi(x) = softmax(l(x)), trained jointly with MSE.
Prediction: mu_hat(x) = sum_k pi_k(x) mu_k(x).
"""
import copy

import numpy as np
import torch
from torch import nn


def _mlp(d_in, hidden, d_out, dropout):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class MoENet(nn.Module):
    def __init__(self, d_in, K, hidden=(32, 32, 32), dropout=0.1):
        super().__init__()
        self.K = K
        self.experts = nn.ModuleList([_mlp(d_in, hidden, 1, dropout) for _ in range(K)])
        self.gate = _mlp(d_in, hidden, K, dropout)

    def forward(self, x):
        pi = torch.softmax(self.gate(x), dim=1)                     # (B, K), Eq. (1)
        mus = torch.cat([e(x) for e in self.experts], dim=1)        # (B, K)
        mu = (pi * mus).sum(dim=1)                                  # (B,),  Eq. (2)
        return mu, pi


class MoERegressor:
    """sklearn-style wrapper: fit(X, y) / predict(X) -> (mu, pi)."""

    def __init__(self, K=3, hidden=(32, 32, 32), dropout=0.1, lr=1e-4, batch_size=64,
                 max_epochs=600, val_frac=0.0, patience=100, seed=0):
        self.K, self.hidden, self.dropout = K, tuple(hidden), dropout
        self.lr, self.batch_size, self.max_epochs = lr, batch_size, max_epochs
        self.val_frac, self.patience, self.seed = val_frac, patience, seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
        y = np.asarray(y, dtype=np.float32).ravel()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = MoENet(X.shape[1], self.K, self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # Default (val_frac=0): train max_epochs on the whole training split, as the paper's runtimes
        # imply.  With val_frac > 0, early stopping on a held-out slice of the training split.
        perm = rng.permutation(len(X))
        n_val = int(round(self.val_frac * len(X)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xtr, ytr = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
        Xva, yva = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

        best, best_state, bad = np.inf, None, 0
        self.n_epochs_ = 0
        for epoch in range(self.max_epochs):
            self.net.train()
            order = torch.randperm(len(Xtr))
            for s in range(0, len(Xtr), self.batch_size):
                b = order[s:s + self.batch_size]
                mu, _ = self.net(Xtr[b])
                loss = ((mu - ytr[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.n_epochs_ = epoch + 1
            if n_val == 0:
                continue
            self.net.eval()
            with torch.no_grad():
                v = ((self.net(Xva)[0] - yva) ** 2).mean().item()
            if v < best - 1e-6:
                best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32).reshape(len(X), -1))
        mu, pi = self.net(X)
        return mu.numpy().astype(np.float64), pi.numpy().astype(np.float64)
