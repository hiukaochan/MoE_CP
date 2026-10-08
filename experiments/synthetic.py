"""Sec. 4.1 synthetic experiment (Figs. 2-4): three domains, unbalanced proportions.

python experiments/synthetic.py [--reps 50] [--n_jobs -1] [--setting 1]
"""
import argparse
import os

import numpy as np
from joblib import Parallel, delayed

from common import METHODS, RESULTS, interval_lengths, run_methods
from moecp.data import make_synthetic

SETTINGS = {1: (0.2, 0.3, 0.5), 2: (1 / 3, 1 / 3, 1 / 3)}
MOE = dict(K=3, hidden=(32, 32, 32), max_epochs=600)


def one_rep(rep, props, n, tau, alpha, methods):
    rng = np.random.default_rng(1000 + rep)
    data = {}
    for k in ["train", "cal", "test"]:
        X, y, dom = make_synthetic(n, props, rng)
        data[k] = (X, y)
        if k == "test":
            dom_test = dom
    res, model, fit_time = run_methods(data, MOE, tau, alpha, seed=rep, methods=methods)
    row = dict(rep=rep, epochs=model.n_epochs_ if model is not None else np.nan)
    for m, r in res.items():
        row[f"{m}_coverage"] = r["cov"].mean()
        row.update(interval_lengths(m, r["lo"], r["hi"], r.get("r_max")))
        row[f"{m}_time"] = r["time"]
    detail = None
    if rep == 0:  # keep one representative run for Figs. 3-4
        Xt, yt = data["test"]
        pi = model.predict(Xt)[1] if model is not None else np.zeros((len(Xt), 0))
        detail = dict(X=Xt.ravel(), y=yt, dom=dom_test, pi=pi,
                      **{f"{m}_{k}": res[m][k] for m in res for k in ["lo", "hi", "cov"]})
    return row, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--setting", type=int, default=1, choices=[1, 2])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tau", type=float, default=150)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    a = ap.parse_args()

    out = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(one_rep)(r, SETTINGS[a.setting], a.n, a.tau, a.alpha, a.methods) for r in range(a.reps))
    import pandas as pd
    df = pd.DataFrame([o[0] for o in out])
    os.makedirs(RESULTS, exist_ok=True)
    tag = f"synthetic_setting{a.setting}"
    df.to_csv(os.path.join(RESULTS, f"{tag}.csv"), index=False)
    np.savez(os.path.join(RESULTS, f"{tag}_rep0.npz"), **out[0][1])
    cols = [c for c in df.columns if c.split("_", 1)[-1] in ("coverage", "length", "capped_length", "inf_frac", "time")]
    print(df[cols].describe().loc[["mean", "std", "50%"]].round(4).to_string())


if __name__ == "__main__":
    main()
