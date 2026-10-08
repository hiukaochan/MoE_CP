import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from moecp import MoECP, MoERegressor  # noqa: E402
from moecp.baselines.cc import SCPCC  # noqa: E402
from moecp.baselines.chr import CHR  # noqa: E402
from moecp.baselines.cqr import CQR  # noqa: E402
from moecp.baselines.pcp import PCP  # noqa: E402
from moecp.baselines.qrf import QRF  # noqa: E402
from moecp.baselines.rlcp import RLCP  # noqa: E402
from moecp.metrics import covered  # noqa: E402

METHODS = ["CQR", "CHR", "SCP+CC", "RLCP", "PCP", "MoECP"]   # paper's order
BASE = {"CQR": "qrf", "CHR": "qrf", "SCP+CC": "rf", "RLCP": "rf", "PCP": "rf", "MoECP": "moe"}


def run_methods(splits, moe_kwargs, tau, alpha, seed, methods=METHODS, divergence="kl"):
    """Fit each needed base model once on the training split, then calibrate/evaluate methods.

    Base models (paper App. A.4.1): MoE for MoE-CP; quantile random forest for CQR and CHR;
    sklearn RandomForestRegressor (default settings) for SCP+CC, RLCP and PCP.
    Returns ({method: dict(lo, hi, cov, time)}, moe_model_or_None, {base: fit_seconds}).
    `time` = base-model fit time + the method's own fit/calibrate/predict time.
    """
    torch.set_num_threads(1)
    (Xtr, ytr), (Xc, yc), (Xt, yt) = splits["train"], splits["cal"], splits["test"]
    Xtr2 = np.asarray(Xtr).reshape(len(Xtr), -1)
    needed = {BASE[m] for m in methods}
    bases, fit_time = {}, {}
    for b in sorted(needed):
        t0 = time.time()
        if b == "moe":
            bases[b] = MoERegressor(seed=seed, **moe_kwargs).fit(Xtr, ytr)
        elif b == "qrf":
            bases[b] = QRF(seed=seed).fit(Xtr2, ytr)
        else:
            bases[b] = RandomForestRegressor(random_state=seed, n_jobs=1).fit(Xtr2, np.ravel(ytr))
        fit_time[b] = time.time() - t0

    y_rng = np.ptp(ytr)
    out = {}
    for name in methods:
        t0 = time.time()
        if name == "MoECP":
            cp = MoECP(bases["moe"], tau=tau, divergence=divergence, alpha=alpha, seed=seed)
        elif name == "SCP+CC":
            cp = SCPCC(bases["rf"], alpha=alpha, seed=seed)
        elif name == "CQR":
            cp = CQR(bases["qrf"], alpha=alpha)
        elif name == "CHR":
            cp = CHR(bases["qrf"], alpha=alpha, ymin=np.min(ytr) - 0.1 * y_rng,
                     ymax=np.max(ytr) + 0.1 * y_rng, seed=seed)
        elif name == "RLCP":
            cp = RLCP(bases["rf"], Xtr2, alpha=alpha, n_eff=100, seed=seed)
        elif name == "PCP":
            cp = PCP(bases["rf"], alpha=alpha, seed=seed).train(Xtr2, ytr)
        lo, hi = cp.calibrate(Xc, yc).predict_interval(Xt)
        out[name] = dict(lo=lo, hi=hi, cov=covered(lo, hi, yt),
                         time=time.time() - t0 + fit_time[BASE[name]])
        if not np.all(np.isfinite(hi - lo)):
            # PCP-repo convention (finite=True): an infinite half-width counts as the largest
            # absolute test residual of the method's point predictor.
            mu = bases["moe"].predict(Xt)[0] if BASE[name] == "moe" else bases["rf"].predict(
                np.asarray(Xt).reshape(len(Xt), -1))
            out[name]["r_max"] = np.max(np.abs(np.ravel(yt) - mu))
    return out, bases.get("moe"), fit_time


def interval_lengths(name, lo, hi, r_max=None):
    """Length summaries that stay finite when some intervals are (-inf, inf).

    MoE-CP, RLCP and PCP return an infinite interval when the test point's own weight exceeds
    alpha. Reported: mean over finite intervals, median, fraction infinite, and the mean with
    infinite intervals capped at 2 * r_max (largest absolute test residual; PCP-repo convention).
    SCP+CC calibrates its two bounds separately, so they can cross; such a point counts as not
    covered and its length as |hi - lo| (also the PCP-repo convention).
    """
    L = np.abs(hi - lo)
    fin = np.isfinite(L)
    capped = np.where(fin, L, 2 * r_max if r_max is not None else np.inf)
    return {f"{name}_length": L[fin].mean() if fin.any() else np.inf,
            f"{name}_capped_length": capped.mean(),
            f"{name}_median_length": np.median(L),
            f"{name}_inf_frac": 1 - fin.mean()}
