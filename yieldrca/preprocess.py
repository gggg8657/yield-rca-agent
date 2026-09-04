"""Fold-internal cleaning for SECOM-shaped sensor matrices.

Everything here is a scikit-learn transformer with a real ``fit`` so it can sit
*inside* a cross-validation pipeline. That is not a style choice: SECOM's
classic evaluation mistake is deciding which columns are constant, which are
duplicates, and what the imputation medians are while looking at the whole
dataset -- all three of those leak test-fold information into training. Fitted
per fold, they cannot.

The drop rules, in order:

1. **all-missing in this fold** -- nothing to impute from.
2. **constant** -- zero variance over the fold's observed values; carries no
   information and makes standardisation ill-posed.
3. **too sparse** -- observed fraction below ``min_observed`` (default: keep
   everything; SECOM's worst sensor is 91% missing, which we keep and let the
   missing-indicator carry).
4. **exact duplicates** -- identical value *and* NaN pattern within the fold.
   SECOM has 7 such groups covering 104 removable columns; leaving them in
   splits one physical signal's importance across several identical names and
   corrupts any top-k stability measurement.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SensorCleaner(BaseEstimator, TransformerMixin):
    """Drop all-missing, constant, over-sparse and duplicated sensor columns.

    After ``fit``, ``keep_`` holds the surviving column indices *into the
    original matrix*, so any importance ranking can be mapped straight back to
    sensor names, and ``dropped_`` records why each column went.
    """

    def __init__(self, min_observed: float = 0.0, dedupe: bool = True):
        self.min_observed = min_observed
        self.dedupe = dedupe

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        nan = np.isnan(X)
        obs = 1.0 - nan.mean(axis=0)

        reason = np.full(p, "", dtype=object)
        drop = np.zeros(p, dtype=bool)

        all_missing = obs == 0.0
        reason[all_missing & ~drop] = "all_missing"
        drop |= all_missing

        # only ask for min/max where at least one value is observed, so the
        # all-missing columns just dropped do not raise an All-NaN warning
        lo = np.zeros(p)
        hi = np.zeros(p)
        live = np.flatnonzero(~drop)
        if len(live):
            sub = np.where(nan[:, live], np.nan, X[:, live])
            lo[live] = np.nanmin(sub, axis=0)
            hi[live] = np.nanmax(sub, axis=0)
        const = ~drop & (lo == hi)
        reason[const] = "constant"
        drop |= const

        sparse = ~drop & (obs < self.min_observed)
        reason[sparse] = "too_sparse"
        drop |= sparse

        dup_groups: list[list[int]] = []
        if self.dedupe:
            keys: dict[bytes, list[int]] = {}
            for j in np.flatnonzero(~drop):
                k = np.nan_to_num(X[:, j], nan=-1.2345e30).tobytes()
                keys.setdefault(k, []).append(int(j))
            for g in keys.values():
                if len(g) > 1:
                    dup_groups.append(g)
                    for j in g[1:]:
                        drop[j] = True
                        reason[j] = f"duplicate_of_{g[0]}"

        self.n_features_in_ = p
        self.keep_ = np.flatnonzero(~drop)
        self.dropped_ = {int(j): str(reason[j]) for j in np.flatnonzero(drop)}
        self.duplicate_groups_ = dup_groups
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, self.keep_]

    def drop_counts(self):
        out: dict[str, int] = {}
        for r in self.dropped_.values():
            key = "duplicate" if r.startswith("duplicate_of_") else r
            out[key] = out.get(key, 0) + 1
        return out


class MissingIndicatorAppender(BaseEstimator, TransformerMixin):
    """Append a 0/1 missingness column for sensors that are missing in-fold.

    In a fab, *whether* a metrology step reported a value is itself a signal
    (a skipped measurement, a tool down). Median imputation destroys it, so it
    is re-attached explicitly rather than hoped for.
    """

    def __init__(self, min_frac: float = 0.01):
        self.min_frac = min_frac

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        frac = np.isnan(X).mean(axis=0)
        self.cols_ = np.flatnonzero(frac >= self.min_frac)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if len(self.cols_) == 0:
            return X
        return np.hstack([X, np.isnan(X[:, self.cols_]).astype(np.float64)])


class UnivariateTopK(BaseEstimator, TransformerMixin):
    """Keep the ``k`` columns with the largest |per-sensor AUC - 0.5|.

    The naive-selection control. If the agent loop cannot beat this at the same
    sparsity, the plan/verify machinery is not earning its keep.
    """

    def __init__(self, k: int = 25):
        self.k = k

    def fit(self, X, y):
        from sklearn.metrics import roc_auc_score

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        med = np.nanmedian(np.where(np.isnan(X), np.nan, X), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xi = np.where(np.isnan(X), med, X)
        sc = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            if np.ptp(Xi[:, j]) > 0:
                sc[j] = abs(roc_auc_score(y, Xi[:, j]) - 0.5)
        self.keep_ = np.sort(np.argsort(sc)[::-1][: min(self.k, X.shape[1])])
        self.scores_ = sc
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.keep_]
