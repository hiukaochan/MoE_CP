"""Quantile random forest black box shared by CQR and CHR (as in the CHR reference code)."""
import numpy as np
from quantile_forest import RandomForestQuantileRegressor


class QRF:
    def __init__(self, quantiles=np.round(np.arange(0.01, 1.0, 0.01), 2), n_estimators=100,
                 min_samples_leaf=50, seed=0):
        self.quantiles = np.sort(np.asarray(quantiles))
        self.model = RandomForestQuantileRegressor(n_estimators=n_estimators,
                                                   min_samples_leaf=min_samples_leaf,
                                                   random_state=seed, n_jobs=1)

    def fit(self, X, y):
        self.model.fit(np.asarray(X).reshape(len(X), -1), np.ravel(y))
        return self

    def predict(self, X):
        """(n, len(quantiles)) conditional quantiles, sorted row-wise to remove crossings."""
        q = self.model.predict(np.asarray(X).reshape(len(X), -1), quantiles=list(self.quantiles))
        return np.sort(np.asarray(q).reshape(len(X), -1), axis=1)
