"""Orchestrator for the dependency-free path: Sensor -> Correlator -> Verifier -> Reporter.

This is the pure-NumPy version of the loop, kept so the repo runs with nothing
but numpy installed. It mirrors the sklearn-backed :class:`yieldrca.estimator.AgentRCA`
in structure but not in rigour -- for any measurement you intend to publish,
use ``AgentRCA`` inside a cross-validation harness (``yieldrca.evaluate``).

One thing it does *not* do is report an in-sample AUC. Fitting on all rows and
scoring the same rows on 200+ sensors reads far above 0.9 on data with almost
no learnable signal, which makes it worse than useless as a headline. ``auc``
here is a 5-fold stratified cross-validated AUC with the model refitted per
fold; ``auc_in_sample`` is kept beside it, explicitly named, to show the size
of the gap.
"""
from __future__ import annotations

import numpy as np

from .agents import CorrelatorAgent, ReporterAgent, SensorAgent, VerifierAgent
from .model import LogisticRCA, roc_auc


def stratified_folds(y, n_splits=5, seed=0):
    """Stratified k-fold index lists, numpy only."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % n_splits].append(int(j))
    out = []
    allidx = np.arange(len(y))
    for f in folds:
        te = np.array(sorted(f), dtype=int)
        out.append((np.setdiff1d(allidx, te), te))
    return out


def cv_auc(model_ctor, X, y, n_splits=5, seed=0):
    """Cross-validated AUC: the model is refitted on each training fold."""
    scores = []
    for tr, te in stratified_folds(y, n_splits, seed):
        if y[tr].sum() < 2 or y[te].sum() < 1:
            continue
        m = model_ctor().fit(X[tr], y[tr])
        scores.append(roc_auc(y[te], m.predict_proba(X[te])))
    return (float(np.mean(scores)) if scores else float("nan"),
            [float(s) for s in scores])


def run_rca(X, y, names, model_ctor=LogisticRCA, top_k=8, stability_min=0.5,
            n_splits=5, seed=0):
    """Run the agent loop and return suspects, groups, stability and AUC."""
    auc, fold_aucs = cv_auc(model_ctor, X, y, n_splits=n_splits, seed=seed)

    model = model_ctor().fit(X, y)
    auc_in = roc_auc(y, model.predict_proba(X))

    sensor = SensorAgent(model)
    ranked = sensor.attribute(X, y, top_k=top_k)
    idx = [i for i, _ in ranked]

    verifier = VerifierAgent()
    stab = verifier.stability(model_ctor, X, y, idx, top_k=top_k, seed=seed)

    # Verify-and-drop loop: keep only stable suspects
    survivors = [(i, s) for (i, s) in ranked if stab.get(i, 0) >= stability_min]
    if not survivors:  # never return empty-handed
        survivors = ranked[:3]
    surv_idx = [i for i, _ in survivors]

    corr = CorrelatorAgent()
    clusters = corr.cluster(X, surv_idx)

    report = ReporterAgent().write(survivors, clusters, stab, names)
    return {
        "auc": auc,
        "auc_folds": fold_aucs,
        "auc_in_sample": float(auc_in),
        "ranked": survivors,
        "clusters": clusters,
        "stability": stab,
        "report": report,
    }
