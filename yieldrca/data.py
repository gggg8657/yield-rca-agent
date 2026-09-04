"""Data loading for semiconductor yield RCA.

Two paths, kept deliberately separate because they answer different questions:

* **Real** — UCI SECOM (1,567 wafers x 590 sensors, pass/fail). No ground-truth
  root causes exist, so this path can only ever measure *prediction* and
  *selection stability*, never "causal sensors recovered".
* **Synthetic** — a generator with the same problem *shape* (high dimensionality,
  heavy class imbalance, missing values, causal signal buried in correlated
  noise) where the causal set is known by construction. This is the only place a
  recovery claim can be made.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SECOM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
SECOM_LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

DEFAULT_ROOT = "data"


def _fetch(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    xp, yp = root / "secom.data", root / "secom_labels.data"
    if not (xp.exists() and yp.exists()):
        import urllib.request

        urllib.request.urlretrieve(SECOM_URL, xp)
        urllib.request.urlretrieve(SECOM_LABELS_URL, yp)
    return xp, yp


def load_secom(root: str = DEFAULT_ROOT, with_time: bool = False):
    """Load UCI SECOM from ``root`` (downloading only if absent).

    Returns ``(X, y, names)``, or ``(X, y, names, t)`` with ``with_time=True``
    where ``t`` is a float64 array of Unix seconds from the label file's
    timestamp column.

    ``y``: 1 = fail (the minority class, ~6.6%), 0 = pass. Missing sensor
    readings are kept as ``NaN`` -- imputation belongs inside the CV fold, not
    here (see :mod:`yieldrca.preprocess`).
    """
    import pandas as pd

    xp, yp = _fetch(Path(root))
    X = pd.read_csv(xp, sep=r"\s+", header=None).values.astype(np.float64)
    lab = pd.read_csv(yp, sep=r"\s+", header=None, quotechar='"',
                      names=["y", "stamp"])
    y = (lab["y"].values == 1).astype(np.int64)
    names = [f"sensor_{i:03d}" for i in range(X.shape[1])]
    if not with_time:
        return X, y, names
    ts = pd.to_datetime(lab["stamp"], format="%d/%m/%Y %H:%M:%S")
    return X, y, names, ts.values.astype("datetime64[s]").astype(np.float64)


def make_synthetic(
    n=1500, p=200, n_causal=5, fail_rate=0.07, missing_rate=0.04, seed=0
):
    """Synthetic yield data: ``p`` sensors, only ``n_causal`` drive the label.

    Mirrors SECOM's pain points: high-dim, imbalanced, missing values, causal
    signal buried in block-correlated noise so raw correlation alone cannot
    find it. Returns ``(X, y, names, causal_idx)``.
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


def secom_profile(X, y):
    """Descriptive stats used by the README/RESULTS generator (no modelling)."""
    nan = np.isnan(X)
    nan_frac = nan.mean(axis=0)
    n_unique = np.array(
        [len(np.unique(X[~nan[:, j], j])) for j in range(X.shape[1])]
    )
    # exact-duplicate sensor columns (NaN pattern included in the key)
    keys: dict[bytes, list[int]] = {}
    for j in range(X.shape[1]):
        k = np.nan_to_num(X[:, j], nan=-1.2345e30).tobytes()
        keys.setdefault(k, []).append(j)
    dup_groups = [g for g in keys.values() if len(g) > 1]
    return {
        "n_wafers": int(X.shape[0]),
        "n_sensors": int(X.shape[1]),
        "n_fail": int(y.sum()),
        "fail_rate": float(y.mean()),
        "imbalance_ratio": float((len(y) - y.sum()) / max(y.sum(), 1)),
        "missing_frac_overall": float(nan.mean()),
        "sensors_with_any_missing": int((nan_frac > 0).sum()),
        "sensors_missing_gt_20pct": int((nan_frac > 0.20).sum()),
        "sensors_missing_gt_50pct": int((nan_frac > 0.50).sum()),
        "sensors_all_missing": int((nan_frac == 1.0).sum()),
        "rows_with_any_missing": int(nan.any(axis=1).sum()),
        "max_missing_frac": float(nan_frac.max()),
        "constant_sensors": int((n_unique <= 1).sum()),
        "duplicate_groups": len(dup_groups),
        "duplicate_sensors_removable": int(sum(len(g) - 1 for g in dup_groups)),
    }
