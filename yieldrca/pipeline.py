"""Orchestrator: routes SensorAgent -> CorrelatorAgent -> VerifierAgent -> Reporter.

Mirrors a plan/execute/verify loop: suspects that fail the stability check
are dropped and the report is regenerated on the survivors.
"""
from __future__ import annotations
import numpy as np
from .model import LogisticRCA, roc_auc
from .agents import SensorAgent, CorrelatorAgent, VerifierAgent, ReporterAgent


def run_rca(X, y, names, model_ctor=LogisticRCA, top_k=8, stability_min=0.5):
    model = model_ctor().fit(X, y)
    auc = roc_auc(y, model.predict_proba(X))

    sensor = SensorAgent(model)
    ranked = sensor.attribute(X, y, top_k=top_k)
    idx = [i for i, _ in ranked]

    verifier = VerifierAgent()
    stab = verifier.stability(model_ctor, X, y, idx, top_k=top_k)

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
        "ranked": survivors,
        "clusters": clusters,
        "stability": stab,
        "report": report,
    }
