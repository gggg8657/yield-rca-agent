"""Measure the top-5 stability KPI on SECOM, for the loop and for the baselines.

    OMP_NUM_THREADS=1 python scripts/stability_secom.py --boot 200 --jobs 16

The metric definitions live in ``yieldrca/stability.py`` and were written down
before this ran. Two resampling schemes are reported:

* **bootstrap** -- ``--boot`` resamples of the wafer set with replacement,
* **cv_train** -- the 25 training folds of the same repeated stratified CV the
  AUC table uses, so the stability number and the AUC number describe the same
  splits.

Every ranker is re-derived from scratch on each replicate, and each is also
scored against the random-ranker floor (5 / 474 ~ 1%).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from arms import AGENT_BASE_KW, AGENT_CFG, SEED
from yieldrca.attribution import (
    permutation_importance_heldout,
    screen_model,
    screen_multivariate,
    screen_univariate,
)
from yieldrca.data import load_secom
from yieldrca.estimator import AgentRCA, make_rf
from yieldrca.preprocess import SensorCleaner
from yieldrca.stability import (
    bootstrap_replicates,
    cluster_map,
    cv_train_replicates,
    measure,
    random_floor,
)


def _cleaned(X, y):
    cl = SensorCleaner().fit(X)
    return cl, cl.transform(X)


def rank_rf_impurity(X, y):
    cl, Xc = _cleaned(X, y)
    _, w = screen_model(lambda a, b: make_rf(n_estimators=500, min_samples_leaf=5)
                        .fit(a, b), Xc, y, n_keep=1)
    return cl.keep_[np.argsort(w)[::-1]]


def rank_logreg_coef(X, y):
    cl, Xc = _cleaned(X, y)
    _, w = screen_multivariate(Xc, y, n_keep=1, C=0.03)
    return cl.keep_[np.argsort(w)[::-1]]


def rank_univariate(X, y):
    cl, Xc = _cleaned(X, y)
    _, w = screen_univariate(Xc, y, n_keep=1)
    return cl.keep_[np.argsort(w)[::-1]]


def rank_perm_only(X, y):
    """SensorAgent alone: screen + held-out permutation, no correlate/verify."""
    cl, Xc = _cleaned(X, y)
    cand, _ = screen_model(lambda a, b: make_rf(n_estimators=300).fit(a, b),
                           Xc, y, n_keep=AGENT_CFG["n_screen"])
    imp = permutation_importance_heldout(
        lambda a, b: make_rf(n_estimators=300).fit(a, b), Xc, y, cand,
        n_splits=AGENT_CFG["n_inner"], n_repeats=AGENT_CFG["n_repeats"], seed=SEED)
    return cl.keep_[np.argsort(imp)[::-1]]


def rank_agent(X, y):
    """Full loop: attribute -> correlate -> verify -> drop, as reported."""
    return AgentRCA(base="rf", base_kw=AGENT_BASE_KW, **AGENT_CFG) \
        .fit(X, y).ranking()


def rank_agent_no_corr(X, y):
    """Ablation: same loop with correlation grouping switched off."""
    cfg = dict(AGENT_CFG)
    cfg["corr_thresh"] = 1.01  # no two sensors ever group
    return AgentRCA(base="rf", base_kw=AGENT_BASE_KW, **cfg).fit(X, y).ranking()


def rank_agent_model(X, y):
    """The full loop with model-native attribution instead of permutation.

    Everything else is the pre-registered operating point. Three separate
    measurements now point at held-out permutation importance as the loop's
    noisy component -- it loses to model-native importance at every depth in
    the sensitivity sweep, the permutation-based rankers are the least stable
    in this very table, and its bootstrap support separates a real world from a
    permuted one worse than a univariate ranker's does. If that diagnosis is
    right, swapping the attribution statistic and changing nothing else should
    move this number; if the sample-size wall is what binds, it should not.
    """
    cfg = dict(AGENT_CFG)
    cfg["attribution"] = "model"
    return AgentRCA(base="rf", base_kw=AGENT_BASE_KW, **cfg).fit(X, y).ranking()


RANKERS = {
    "univariate": ("per-sensor |AUC - 0.5|", rank_univariate),
    "logreg_coef": ("|standardised logistic coefficient|", rank_logreg_coef),
    "rf_impurity": ("random-forest impurity importance", rank_rf_impurity),
    "perm_only": ("SensorAgent only: screen + held-out permutation AUC drop",
                  rank_perm_only),
    "agent_no_corr": ("attribute -> verify -> drop, correlation grouping off",
                      rank_agent_no_corr),
    "agent": ("full agent loop: attribute -> correlate -> verify -> drop",
              rank_agent),
    "agent_model": ("full loop, model-native attribution instead of permutation",
                    rank_agent_model),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cluster-thresh", type=float, default=0.99)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None,
                    help="measure just these rankers (keys of RANKERS)")
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing --out rather than starting "
                         "fresh; replicates and seeds are unchanged, so the "
                         "rows stay comparable")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/secom_stability.json")
    a = ap.parse_args()

    X, y, names = load_secom(a.root)
    cl = SensorCleaner().fit(X)
    p_eff = int(len(cl.keep_))
    cmap, n_multi = cluster_map(X, thresh=a.cluster_thresh)
    n_clusters_eff = len({cmap[int(j)] for j in cl.keep_})

    schemes = {
        "bootstrap": bootstrap_replicates(y, n_boot=a.boot, seed=SEED),
        "cv_train": cv_train_replicates(X, y, a.splits, a.repeats, SEED),
    }
    todo = a.only or list(RANKERS)
    prev = {}
    if a.append and Path(a.out).exists():
        prev = json.loads(Path(a.out).read_text())
    out = {
        "definition_module": "yieldrca/stability.py",
        "k": a.k,
        "cluster_thresh": a.cluster_thresh,
        "effective_sensors": p_eff,
        "effective_clusters": n_clusters_eff,
        "random_floor_raw": random_floor(p_eff, a.k),
        "random_floor_cluster": random_floor(n_clusters_eff, a.k),
        "schemes": {k: {"n_replicates": len(v)} for k, v in schemes.items()},
        "rankers": {k: v for k, v in prev.get("rankers", {}).items()
                    if k not in todo},
    }
    for name in todo:
        label, fn = RANKERS[name]
        out["rankers"][name] = {"label": label}
        for scheme, reps in schemes.items():
            t0 = time.time()
            r = measure(fn, X, y, reps, k=a.k, cmap=cmap, n_jobs=a.jobs)
            r["wall_min"] = (time.time() - t0) / 60.0
            out["rankers"][name][scheme] = r
            print(f"{name:<16} {scheme:<10} pairwise {r['raw']['pairwise_overlap']:.3f} "
                  f"cluster {r['cluster']['pairwise_overlap']:.3f} "
                  f"consensus {r['raw']['consensus_freq']:.3f} "
                  f"({r['wall_min']:.1f} min, B={r['n_replicates']})")
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    # keep the table in RANKERS order regardless of --only / --append
    out["rankers"] = {k: out["rankers"][k] for k in RANKERS
                      if k in out["rankers"]}
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
