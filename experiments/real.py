"""Sec. 4.2 real-data experiment (Fig. 5): Bike-sharing and Temperature.

python experiments/real.py --dataset {bike,temperature} [--reps 50] [--n_jobs -1]
"""
import argparse
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import METHODS, RESULTS, interval_lengths, run_methods
from moecp.data import LOADERS, split
from moecp.metrics import worst_slice_coverage

MOE = dict(K=2, hidden=(64, 64, 64), max_epochs=2000)


def one_rep(rep, X, y, n, tau, alpha, methods):
    data = split(X, y, (n, n, n), rng=2000 + rep)
    res, model, fit_time = run_methods(data, MOE, tau, alpha, seed=rep, methods=methods)
    Xt = data["test"][0]
    row = dict(rep=rep, epochs=model.n_epochs_ if model is not None else np.nan)
    for m, r in res.items():
        row[f"{m}_coverage"] = r["cov"].mean()
        row.update(interval_lengths(m, r["lo"], r["hi"], r.get("r_max")))
        row[f"{m}_wsc"] = worst_slice_coverage(Xt, r["cov"], rng=rep)
        row[f"{m}_time"] = r["time"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--tau", type=float, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    a = ap.parse_args()

    X, y = LOADERS[a.dataset]()
    print(f"{a.dataset}: X {X.shape}, y {y.shape}")
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(one_rep)(r, X, y, a.n, a.tau, a.alpha, a.methods) for r in range(a.reps))
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, f"{a.dataset}.csv"), index=False)
    cols = [c for c in df.columns if c.split("_", 1)[-1] in ("coverage", "length", "capped_length", "wsc", "inf_frac", "time")]
    print(df[cols].describe().loc[["mean", "std", "50%"]].round(4).to_string())


if __name__ == "__main__":
    main()
