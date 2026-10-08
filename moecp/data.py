"""Datasets: three-domain synthetic data (Sec. 4.1), Bike-sharing and Temperature (Sec. 4.2)."""
import io
import os
import ssl
import subprocess
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
TEMP_URL = ("https://archive.ics.uci.edu/static/public/514/"
            "bias+correction+of+numerical+prediction+model+temperature+forecast.zip")


def make_synthetic(n, props=(0.2, 0.3, 0.5), rng=None):
    """Y = X + e | X^2 + e | X^3 + e with domain proportions `props`; X ~ U[-3, 3], e ~ N(0, 1)."""
    rng = np.random.default_rng(rng)
    X = rng.uniform(-3, 3, size=n)
    dom = rng.choice(3, size=n, p=np.asarray(props) / np.sum(props))
    f = np.stack([X, X ** 2, X ** 3], axis=1)[np.arange(n), dom]
    y = f + rng.standard_normal(n)
    return X.reshape(-1, 1), y, dom


def _fetch_zip(url, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"Downloading {url}")
        try:  # python.org builds on macOS ship without a CA bundle; use certifi's if present
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = None
        try:
            with urllib.request.urlopen(url, context=ctx) as r, open(path, "wb") as f:
                f.write(r.read())
        except urllib.error.URLError:
            subprocess.run(["curl", "-fsSL", "-o", path, url], check=True)
    return zipfile.ZipFile(path)


def load_bike():
    """UCI Bike Sharing (hour.csv). Response = total rental count `cnt`.

    The 13 features are every column except the date string, the target and its two components
    (`casual` + `registered` = `cnt`).
    """
    z = _fetch_zip(BIKE_URL, "bike_sharing.zip")
    df = pd.read_csv(io.BytesIO(z.read("hour.csv")))
    feats = ["instant", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
             "weathersit", "temp", "atemp", "hum", "windspeed"]
    return df[feats].to_numpy(np.float64), df["cnt"].to_numpy(np.float64)


def load_temperature():
    """UCI LDAPS bias-correction data. Response = Next_Tmin; drop station, Date, Next_Tmax."""
    z = _fetch_zip(TEMP_URL, "temperature.zip")
    csv = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    df = pd.read_csv(io.BytesIO(z.read(csv)), encoding="latin-1")
    df = df.drop(columns=["station", "Date", "Next_Tmax"]).dropna()
    y = df.pop("Next_Tmin").to_numpy(np.float64)
    return df.to_numpy(np.float64), y


LOADERS = {"bike": load_bike, "temperature": load_temperature}


def split(X, y, sizes, rng, standardize=True):
    """Random disjoint subsamples of sizes (n_train, n_cal, n_test).

    With `standardize`, X and y are centered/scaled using training-split statistics.
    Returns dict with keys train/cal/test -> (X, y).
    """
    rng = np.random.default_rng(rng)
    n_tr, n_cal, n_te = sizes
    idx = rng.permutation(len(X))[:n_tr + n_cal + n_te]
    parts = dict(train=idx[:n_tr], cal=idx[n_tr:n_tr + n_cal], test=idx[n_tr + n_cal:])
    out = {k: (X[v].copy(), y[v].copy()) for k, v in parts.items()}
    if standardize:
        Xtr, ytr = out["train"]
        mx, sx = Xtr.mean(0), Xtr.std(0)
        sx[sx == 0] = 1.0
        my, sy = ytr.mean(), ytr.std()
        out = {k: ((Xv - mx) / sx, (yv - my) / sy) for k, (Xv, yv) in out.items()}
    return out
