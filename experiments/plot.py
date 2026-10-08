"""Make the main-text figure analogues from results/ into figures/.

Fig 2: synthetic coverage + length boxplots      -> figures/fig2_synthetic_box.png
Fig 3: synthetic kNN local coverage vs X          -> figures/fig3_local_coverage.png
Fig 4: synthetic prediction intervals             -> figures/fig4_intervals.png
Fig 5: real data coverage / length / worst-slice  -> figures/fig5_real.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import FIGURES, RESULTS  # noqa: E402
from moecp.metrics import local_coverage_knn  # noqa: E402

METHODS = ["CQR", "CHR", "SCP+CC", "RLCP", "PCP", "MoECP"]
COLORS = {"CQR": "#ff7f0e", "CHR": "#2ca02c", "SCP+CC": "#8c8c8c", "RLCP": "#9467bd",
          "PCP": "#8c564b", "MoECP": "#1f77b4"}
ALPHA = 0.1


def _box(ax, df, metric, title, ref=None):
    ms = [m for m in METHODS if f"{m}_{metric}" in df]
    data = [df[f"{m}_{metric}"].to_numpy() for m in ms]
    bp = ax.boxplot(data, tick_labels=ms, patch_artist=True, widths=0.5)
    for patch, m in zip(bp["boxes"], ms):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.6)
    if ref is not None:
        ax.axhline(ref, ls="--", c="r", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Method")


def fig2(tag="synthetic_setting1"):
    df = pd.read_csv(os.path.join(RESULTS, f"{tag}.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _box(axes[0], df, "coverage", "Coverage rate", 1 - ALPHA)
    _box(axes[1], df, "capped_length", "Interval length (inf capped)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_synthetic_box.png"), dpi=150)


def fig3_fig4(tag="synthetic_setting1"):
    d = np.load(os.path.join(RESULTS, f"{tag}_rep0.npz"))
    X, y = d["X"], d["y"]
    order = np.argsort(X)

    ms = [m for m in METHODS if f"{m}_cov" in d]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for m in ms:
        lc = local_coverage_knn(X, d[f"{m}_cov"], k=100)
        ax.plot(X[order], lc[order], label=m, c=COLORS[m])
    ax.axhline(1 - ALPHA, ls="--", c="r", lw=1)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("X")
    ax.set_ylabel("Conditional Coverage")
    ax.legend(ncol=len(ms), fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig3_local_coverage.png"), dpi=150)

    rows = int(np.ceil(len(ms) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.2 * rows), sharey=True, squeeze=False)
    for ax, m in zip(axes.ravel(), ms):
        lo, hi = d[f"{m}_lo"][order], d[f"{m}_hi"][order]
        lo, hi = np.clip(lo, -45, 45), np.clip(hi, -45, 45)  # infinite intervals drawn to the frame
        ax.fill_between(X[order], lo, hi, color=COLORS[m], alpha=0.3, step="mid", lw=0)
        ax.scatter(X, y, s=4, c="k", alpha=0.5)
        ax.set_ylim(-40, 40)
        ax.set_title(f"Prediction Interval for {m}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig4_intervals.png"), dpi=150)


def fig5(datasets=("bike", "temperature")):
    avail = [ds for ds in datasets if os.path.exists(os.path.join(RESULTS, f"{ds}.csv"))]
    if not avail:
        return
    fig, axes = plt.subplots(len(avail), 3, figsize=(16, 3.8 * len(avail)), squeeze=False)
    for row, ds in zip(axes, avail):
        df = pd.read_csv(os.path.join(RESULTS, f"{ds}.csv"))
        _box(row[0], df, "coverage", f"{ds}: coverage rate", 1 - ALPHA)
        _box(row[1], df, "capped_length", f"{ds}: interval length (inf capped)")
        _box(row[2], df, "wsc", f"{ds}: worst-slice coverage", 1 - ALPHA)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig5_real.png"), dpi=150)


def summary():
    rows = []
    for f in sorted(os.listdir(RESULTS)):
        if not f.endswith(".csv") or f == "summary.csv":
            continue
        df = pd.read_csv(os.path.join(RESULTS, f))
        for m in METHODS:
            r = dict(experiment=f[:-4], method=m, reps=len(df))
            if f"{m}_coverage" not in df:
                continue
            for k in ["coverage", "length", "median_length", "wsc", "inf_frac", "time"]:
                if f"{m}_{k}" in df:
                    r[k] = f"{df[f'{m}_{k}'].mean():.3f} ± {df[f'{m}_{k}'].std():.3f}"
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    fig2()
    fig3_fig4()
    fig5()
    summary()
