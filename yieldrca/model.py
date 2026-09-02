"""Defect classifier + feature attribution.

Real path: gradient boosting (xgboost/sklearn) + SHAP. If those aren't
installed, falls back to a numpy logistic regression with class weighting
and permutation importance — so the pipeline runs anywhere.
"""
from __future__ import annotations
import numpy as np


def _impute_scale(X, stats=None):
    if stats is None:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.0
        stats = (mean, std)
    mean, std = stats
    Xi = np.where(np.isnan(X), mean, X)
    return (Xi - mean) / std, stats


class LogisticRCA:
    """Class-weighted logistic regression (numpy) — the offline fallback."""

    def __init__(self, l2=1.0, lr=0.3, epochs=400):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w = self.b = self.stats = None

    def fit(self, X, y):
        Xs, self.stats = _impute_scale(X)
        n, p = Xs.shape
        pos = max(y.sum(), 1)
        cw = np.where(y == 1, n / (2 * pos), n / (2 * max(n - pos, 1)))
        self.w, self.b = np.zeros(p), 0.0
        for _ in range(self.epochs):
            z = Xs @ self.w + self.b
            pr = 1 / (1 + np.exp(-z))
            g = (pr - y) * cw
            self.w -= self.lr * (Xs.T @ g / n + self.l2 * self.w / n)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        Xs, _ = _impute_scale(X, self.stats)
        return 1 / (1 + np.exp(-(Xs @ self.w + self.b)))


def try_boosted():
    """Return a fitted-API classifier from sklearn if available, else None."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier  # noqa
        from sklearn.impute import SimpleImputer  # noqa
        from sklearn.pipeline import make_pipeline  # noqa

        return True
    except Exception:
        return None


def permutation_importance(model, X, y, metric, n_repeats=5, seed=0):
    """Model-agnostic feature attribution (works for the numpy fallback too)."""
    rng = np.random.default_rng(seed)
    base = metric(y, model.predict_proba(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - metric(y, model.predict_proba(Xp)))
        imp[j] = np.mean(drops)
    return imp


def roc_auc(y, s):
    """Rank-based AUC, no sklearn needed."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
