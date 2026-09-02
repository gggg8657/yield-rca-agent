"""Data loading for semiconductor yield RCA.

Real path: UCI SECOM (1567 wafers x 590 sensors, pass/fail labels).
Offline path: a synthetic generator with the same *shape* of problem
(high dimensionality, heavy class imbalance, a few truly-causal sensors
buried in noise) so the whole pipeline runs with no download.
"""
from __future__ import annotations
import numpy as np

SECOM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
SECOM_LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"


def load_secom(cache_dir: str = ".data"):
    """Download + parse UCI SECOM. Returns (X, y, feature_names).

    y: 1 = fail (defect), 0 = pass. Requires network + pandas.
    """
    import os
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    xp = os.path.join(cache_dir, "secom.data")
    yp = os.path.join(cache_dir, "secom_labels.data")
    if not os.path.exists(xp):
        import urllib.request

        urllib.request.urlretrieve(SECOM_URL, xp)
        urllib.request.urlretrieve(SECOM_LABELS_URL, yp)
    X = pd.read_csv(xp, sep=r"\s+", header=None).values.astype(float)
    y_raw = pd.read_csv(yp, sep=r"\s+", header=None)[0].values
    y = (y_raw == 1).astype(int)  # SECOM: 1 = fail
    names = [f"sensor_{i:03d}" for i in range(X.shape[1])]
    return X, y, names


def make_synthetic(
    n=1500, p=200, n_causal=5, fail_rate=0.07, missing_rate=0.04, seed=0
):
    """Synthetic yield data: p sensors, only n_causal drive the fail label.

    Mirrors SECOM's pain points: high-dim, imbalanced, missing values,
    causal signal buried in correlated noise. Returns (X, y, names, causal_idx).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # inject block correlation so RCA can't just pick raw correlation
    for b in range(0, p, 20):
        X[:, b : b + 20] += 0.6 * rng.standard_normal((n, 1))
    causal = rng.choice(p, size=n_causal, replace=False)
    w = rng.uniform(1.2, 2.2, size=n_causal) * rng.choice([-1, 1], size=n_causal)
    logit = X[:, causal] @ w
    thresh = np.quantile(logit, 1 - fail_rate)
    p_fail = 1 / (1 + np.exp(-(logit - thresh) * 1.5))
    y = (rng.uniform(size=n) < p_fail).astype(int)
    mask = rng.uniform(size=X.shape) < missing_rate
    X[mask] = np.nan
    names = [f"sensor_{i:03d}" for i in range(p)]
    return X, y, names, np.sort(causal)
