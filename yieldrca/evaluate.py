"""Cross-validation harness: identical folds for every arm, paired deltas.

The point of this module is that "the agent loop scores 0.7x" is not a claim.
The claim is "the agent loop scores this much *more than the obvious
baseline*", and on 1,567 wafers with 104 fails, fold-to-fold noise is larger
than any plausible effect -- so every arm sees byte-identical train/test
indices and the comparison is made **per fold, paired**. A paired 95% CI that
straddles zero means we cannot tell the arms apart, and that is a result worth
printing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def _t_crit(df, conf=0.95):
    from scipy import stats

    return float(stats.t.ppf(0.5 + conf / 2, df))


def mean_ci(values, conf=0.95):
    """Mean and Student-t CI half-width over folds (n-1 df)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "sd": float("nan"), "n": 0}
    if n == 1:
        return {"mean": float(v[0]), "ci_lo": float(v[0]), "ci_hi": float(v[0]),
                "sd": 0.0, "n": 1}
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = _t_crit(n - 1, conf) * sd / np.sqrt(n)
    return {"mean": m, "ci_lo": m - h, "ci_hi": m + h, "sd": sd, "n": n}


def paired_delta(a, b, conf=0.95):
    """Paired mean difference a - b over folds, with CI and a Wilcoxon p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = mean_ci(d, conf)
    out["wins"] = int((d > 0).sum())
    out["losses"] = int((d < 0).sum())
    try:
        from scipy import stats

        if np.any(d != 0):
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
            out["ttest_p"] = float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        pass
    return out


@dataclass
class Arm:
    """One thing to evaluate: a name, a factory, and what it is for."""

    name: str
    factory: object
    label: str = ""
    kind: str = "baseline"  # baseline | agent | control
    extras: object = None   # fn(fitted_estimator) -> dict, recorded per fold
    meta: dict = field(default_factory=dict)


def _run_fold(arm, X, y, tr, te, fold_id):
    t0 = time.time()
    est = clone(arm.factory()) if callable(arm.factory) else clone(arm.factory)
    est.fit(X[tr], y[tr])
    if hasattr(est, "predict_proba"):
        s = est.predict_proba(X[te])
        s = s[:, 1] if np.ndim(s) == 2 else s
    else:
        s = est.decision_function(X[te])
    rec = {
        "arm": arm.name,
        "fold": fold_id,
        "auc": float(roc_auc_score(y[te], s)),
        "ap": float(average_precision_score(y[te], s)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_fail_test": int(y[te].sum()),
        "fit_s": time.time() - t0,
    }
    if arm.extras is not None:
        rec.update(arm.extras(est))
    return rec


def repeated_cv(arms, X, y, n_splits=5, n_repeats=5, seed=0, n_jobs=16,
                verbose=10):
    """Evaluate every arm on identical repeated-stratified-CV folds.

    Returns ``(records, folds)`` where ``records`` is one dict per (arm, fold).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    folds = list(cv.split(X, y))
    jobs = [(a, i, tr, te) for a in arms for i, (tr, te) in enumerate(folds)]
    recs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_run_fold)(a, X, y, tr, te, i) for a, i, tr, te in jobs)
    return list(recs), folds


def summarize(records, metric="auc", conf=0.95):
    """Per-arm mean/CI plus every arm's paired delta against each other arm."""
    by_arm: dict[str, dict[int, float]] = {}
    for r in records:
        by_arm.setdefault(r["arm"], {})[r["fold"]] = r[metric]
    arms = list(by_arm)
    common = sorted(set.intersection(*(set(v) for v in by_arm.values())))
    summ = {a: mean_ci([by_arm[a][f] for f in common], conf) for a in arms}
    deltas = {}
    for a in arms:
        for b in arms:
            if a == b:
                continue
            deltas[f"{a}__vs__{b}"] = paired_delta(
                [by_arm[a][f] for f in common], [by_arm[b][f] for f in common],
                conf)
    return {"metric": metric, "n_folds": len(common), "per_arm": summ,
            "paired": deltas,
            "per_fold": {a: [by_arm[a][f] for f in common] for a in arms}}


def chronological_split(t, frac_train=0.7):
    """Train on the earliest ``frac_train`` of wafers by timestamp, test on the rest."""
    order = np.argsort(t, kind="stable")
    cut = int(round(frac_train * len(order)))
    return order[:cut], order[cut:]


def rolling_origin_splits(t, n_blocks=5):
    """Rolling-origin (expanding-window) splits over contiguous time blocks.

    Train on blocks ``0..k``, test on block ``k+1``, for k = 0 .. n_blocks-2.
    One chronological split can be one unlucky fortnight; this asks whether the
    degradation reproduces at every origin. Always trains on the past and tests
    on the future, which is the only protocol that answers "would this have
    worked if we had deployed it".
    """
    order = np.argsort(t, kind="stable")
    blocks = np.array_split(order, n_blocks)
    out = []
    for k in range(n_blocks - 1):
        tr = np.concatenate(blocks[: k + 1])
        out.append((tr, blocks[k + 1]))
    return out
