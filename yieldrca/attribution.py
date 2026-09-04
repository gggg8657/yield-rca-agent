"""Attribution primitives: held-out permutation importance and a cheap screen.

Two deliberate choices, both about not fooling ourselves:

* **Permutation importance is scored on data the model did not fit.** Scoring it
  on the training rows measures how much the model *relies* on a column, which
  on 590 columns and 104 positives is mostly a measure of overfitting. Here the
  training fold is split again, the model is fit on the inner-train part and the
  AUC drop is measured on the inner-validation part.
* **A screen runs first.** Permuting all 474 surviving sensors x repeats x inner
  splits against a boosted tree is affordable but wasteful; a single
  L2-regularised logistic fit on the same fold ranks them multivariately for
  ~20 ms, and only its top slice is permuted. The screen sees training rows
  only, so it leaks nothing -- it just costs recall on sensors that matter
  purely through an interaction.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def _pos_score(model, X):
    p = model.predict_proba(X)
    return p[:, 1] if getattr(p, "ndim", 1) == 2 else p


def screen_multivariate(X, y, n_keep=60, C=0.05, seed=0):
    """Rank columns by |standardised logistic coefficient|; return top ``n_keep``.

    Multivariate (unlike a per-sensor t-test), so a sensor that only looks
    informative because it correlates with a real driver is discounted.
    """
    Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                            random_state=seed).fit(Xs, y)
    w = np.abs(lr.coef_.ravel())
    n_keep = min(n_keep, X.shape[1])
    return np.argsort(w)[::-1][:n_keep], w


def screen_model(fit_fn, X, y, n_keep=60):
    """Rank columns by the base model's own internal importance.

    For a forest that is impurity decrease, for a linear model |coef|. One fit,
    all columns, and -- unlike the logistic screen -- it is measured in the same
    hypothesis space the final model uses, which matters when the final model
    is a tree ensemble and the screen is not.
    """
    m = fit_fn(X, y)
    est = m[-1] if hasattr(m, "steps") else m
    if hasattr(est, "feature_importances_"):
        w = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        w = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    else:  # no introspection available -- fall back to the logistic screen
        return screen_multivariate(X, y, n_keep=n_keep)
    w = w[: X.shape[1]]
    n_keep = min(n_keep, X.shape[1])
    return np.argsort(w)[::-1][:n_keep], w


def screen_univariate(X, y, n_keep=60):
    """Rank columns by per-sensor rank AUC |auc - 0.5| (the naive control)."""
    Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    sc = np.zeros(Xi.shape[1])
    for j in range(Xi.shape[1]):
        col = Xi[:, j]
        if np.ptp(col) == 0:
            continue
        sc[j] = abs(roc_auc_score(y, col) - 0.5)
    n_keep = min(n_keep, X.shape[1])
    return np.argsort(sc)[::-1][:n_keep], sc


def permutation_importance_heldout(
    fit_fn, X, y, cols, n_splits=3, n_repeats=3, test_size=0.3, seed=0
):
    """Mean AUC drop from permuting each column, measured on held-out rows.

    ``fit_fn(X_tr, y_tr)`` must return an object with ``predict_proba``.
    Returns an array over ``X``'s columns (zero outside ``cols``).
    """
    cols = np.asarray(cols, dtype=int)
    imp = np.zeros(X.shape[1])
    n_eff = 0
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                      random_state=seed)
    for k, (tr, va) in enumerate(splitter.split(X, y)):
        if y[va].sum() < 2 or y[tr].sum() < 2:
            continue
        m = fit_fn(X[tr], y[tr])
        base = roc_auc_score(y[va], _pos_score(m, X[va]))
        rng = np.random.default_rng(seed + 1000 * k)
        Xv = X[va]
        for j in cols:
            drops = 0.0
            saved = Xv[:, j].copy()
            for _ in range(n_repeats):
                Xv[:, j] = rng.permutation(saved)
                drops += base - roc_auc_score(y[va], _pos_score(m, Xv))
            Xv[:, j] = saved
            imp[j] += drops / n_repeats
        n_eff += 1
    return imp / max(n_eff, 1)


def correlation_clusters(X, cols, thresh=0.9):
    """Greedy single-link grouping of ``cols`` by |Pearson r| >= ``thresh``.

    SECOM has 179 sensors with a >0.99-correlated partner. Two near-identical
    signals trade places between resamples, so a raw top-k stability score
    punishes the pipeline for something physically meaningless. Grouping them
    and reporting the group is the fix -- and the group is also the more useful
    answer for an engineer, who gets a signal *family* to go look at.
    """
    cols = np.asarray(cols, dtype=int)
    if len(cols) == 0:
        return []
    Xi = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
        X[:, cols])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.nan_to_num(np.corrcoef(Xi.T), nan=0.0)
    C = np.atleast_2d(C)
    adj = np.abs(C) >= thresh
    seen = np.zeros(len(cols), dtype=bool)
    groups = []
    for a in range(len(cols)):
        if seen[a]:
            continue
        stack, comp = [a], []
        seen[a] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(adj[u] & ~seen):
                seen[v] = True
                stack.append(v)
        groups.append([int(cols[i]) for i in sorted(comp)])
    return groups

