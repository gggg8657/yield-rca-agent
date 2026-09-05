"""H8: is the attribution effect on AUC a sparsity artifact?

`agent_model_rf` beats `agent_rf` by +0.0116 AUC, but it selects 25.0 sensors
per fold against 19.8 -- and 25.0 is `max_select` exactly, so it is pinned at
its budget. Selection costs AUC monotonically on SECOM, so the two arms are not
matched on the one thing this repository has shown matters most.

This sweeps `max_select` for **both** attribution statistics under the headline
25-fold protocol, changing nothing else from the pre-registered operating
point, so the two curves can be compared at matched selected-set size instead
of at matched configuration.

**How to read it, decided before the run** (`critique_log.md`, Turn 12). One
rung cannot resolve this: the paired CI half-width here is about 0.012, wider
than the effect under test. The evidence is the *sign pattern across matched
rungs*, reported as a count. The rungs share folds and nest, so they are not
pooled into a single interval -- that would manufacture precision. Per-rung
paired CIs are reported and most are expected to straddle zero.

A rung counts as **matched** when both arms are pinned at the cap, since
`max_select` only binds when the loop's survivor count exceeds it. Actual
per-fold `n_selected` is recorded so matching is verified rather than assumed.

    OMP_NUM_THREADS=1 python scripts/eval_sparsity.py --jobs 16
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from arms import AGENT_BASE_KW, AGENT_CFG, SEED
from yieldrca.data import load_secom
from yieldrca.estimator import AgentRCA
from yieldrca.evaluate import mean_ci, paired_delta

ATTRIBUTIONS = ("permutation", "model")


def _est(attribution, max_select):
    """Pre-registered operating point with two fields moved, nothing else."""
    cfg = dict(AGENT_CFG)
    cfg["attribution"] = attribution
    cfg["max_select"] = int(max_select)
    return AgentRCA(base="rf", base_kw=AGENT_BASE_KW, **cfg)


def _one(X, y, tr, te, fold, attribution, max_select):
    est = clone(_est(attribution, max_select)).fit(X[tr], y[tr])
    p = est.predict_proba(X[te])[:, 1]
    return {
        "fold": fold, "attribution": attribution, "max_select": int(max_select),
        "auc": float(roc_auc_score(y[te], p)),
        "ap": float(average_precision_score(y[te], p)),
        "n_selected": int(len(est.selected_)),
        "n_candidates": int(est.n_candidates_),
        # `pinned` is what makes a rung comparable: the cap bound, so the arm
        # took exactly its budget rather than fewer.
        "pinned": bool(len(est.selected_) >= int(max_select)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-select", default="5,10,15,20,25,40")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/sparsity.json")
    a = ap.parse_args()

    caps = [int(v) for v in a.max_select.split(",")]
    X, y, _ = load_secom(a.root)
    cv = RepeatedStratifiedKFold(n_splits=a.splits, n_repeats=a.repeats,
                                 random_state=SEED)
    folds = list(cv.split(X, y))
    jobs = [(i, tr, te, at, c) for i, (tr, te) in enumerate(folds)
            for at in ATTRIBUTIONS for c in caps]
    print(f"[sparsity] {len(caps)} caps x {len(ATTRIBUTIONS)} statistics x "
          f"{len(folds)} folds = {len(jobs)} fits, {a.jobs} workers", flush=True)

    t0 = time.time()
    recs = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(_one)(X, y, tr, te, i, at, c) for i, tr, te, at, c in jobs)
    wall = time.time() - t0
    print(f"[sparsity] {len(recs)} fits in {wall/60:.1f} min", flush=True)

    def sub(at, c):
        return sorted((r for r in recs
                       if r["attribution"] == at and r["max_select"] == c),
                      key=lambda r: r["fold"])

    curves = {}
    for at in ATTRIBUTIONS:
        curves[at] = {}
        for c in caps:
            rs = sub(at, c)
            curves[at][str(c)] = {
                "max_select": c,
                "auc": mean_ci([r["auc"] for r in rs]),
                "ap": mean_ci([r["ap"] for r in rs]),
                "n_selected_mean": float(np.mean([r["n_selected"] for r in rs])),
                "n_selected_min": int(min(r["n_selected"] for r in rs)),
                "n_selected_max": int(max(r["n_selected"] for r in rs)),
                "pinned_rate": float(np.mean([r["pinned"] for r in rs])),
            }
            print(f"  {at:12s} cap {c:3d}  AUC "
                  f"{curves[at][str(c)]['auc']['mean']:.4f}  n_sel "
                  f"{curves[at][str(c)]['n_selected_mean']:5.1f}  pinned "
                  f"{100*curves[at][str(c)]['pinned_rate']:5.1f}%", flush=True)

    # Matched-rung comparison: model minus permutation, paired per fold, at
    # each cap. A rung is "matched" only where both arms are pinned in every
    # fold -- otherwise the caps agree but the selected-set sizes do not.
    matched, per_rung = [], {}
    for c in caps:
        pm, mm = sub("permutation", c), sub("model", c)
        if not (pm and mm) or [r["fold"] for r in pm] != [r["fold"] for r in mm]:
            continue
        d = paired_delta([r["auc"] for r in mm], [r["auc"] for r in pm])
        pcurve = curves["permutation"][str(c)]
        mcurve = curves["model"][str(c)]
        is_matched = (pcurve["pinned_rate"] == 1.0
                      and mcurve["pinned_rate"] == 1.0)
        per_rung[str(c)] = {
            "max_select": c, "model_minus_permutation": d,
            "n_selected_permutation": pcurve["n_selected_mean"],
            "n_selected_model": mcurve["n_selected_mean"],
            "both_pinned": is_matched,
        }
        if is_matched:
            matched.append((c, d["mean"]))
        print(f"  cap {c:3d}  model - permutation {d['mean']:+.4f} "
              f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
              f"n {mcurve['n_selected_mean']:.1f} vs "
              f"{pcurve['n_selected_mean']:.1f}  "
              f"{'MATCHED' if is_matched else 'unmatched'}", flush=True)

    # The pre-registered read-out: a sign count over matched rungs, not a
    # pooled interval.
    sign = {
        "n_matched_rungs": len(matched),
        "n_model_above": int(sum(1 for _, m in matched if m > 0)),
        "matched_deltas": {str(c): float(m) for c, m in matched},
        "median_matched_delta": (float(np.median([m for _, m in matched]))
                                 if matched else float("nan")),
    }

    # Sparsity price: the slope of each curve where the cap binds, from the
    # pinned rungs only, since unpinned rungs are not on the sparsity axis.
    slopes = {}
    for at in ATTRIBUTIONS:
        pts = [(curves[at][str(c)]["n_selected_mean"],
                curves[at][str(c)]["auc"]["mean"])
               for c in caps if curves[at][str(c)]["pinned_rate"] == 1.0]
        if len(pts) >= 2:
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            slope, intercept = np.polyfit(xs, ys, 1)
            slopes[at] = {"auc_per_sensor": float(slope),
                          "intercept": float(intercept),
                          "n_points": len(pts),
                          "n_range": [float(xs.min()), float(xs.max())]}

    out = {
        "protocol": {
            "hypothesis": "H8. The attribution effect on AUC is not a "
                          "sparsity artifact: the model-native curve lies "
                          "above the permutation curve at matched "
                          "selected-set size.",
            "cv": f"RepeatedStratifiedKFold({a.splits} x {a.repeats}, seed "
                  f"{SEED}) = {len(folds)} folds, identical folds for every "
                  f"arm; the same protocol as runs/secom_eval.json",
            "varied": ["attribution", "max_select"],
            "held_fixed": {k: v for k, v in AGENT_CFG.items()
                           if k not in ("attribution", "max_select")},
            "readout": "sign count over matched rungs, decided before the run "
                       "(critique_log.md Turn 12). Rungs share folds and nest, "
                       "so per-rung paired CIs are reported but NOT pooled -- "
                       "pooling them would manufacture precision.",
            "matched_definition": "both arms pinned at the cap in every fold, "
                                  "verified from per-fold n_selected rather "
                                  "than assumed from the cap",
            "leakage_control": "the loop is an sklearn estimator; cleaning, "
                               "screening, attribution and verification are "
                               "fitted inside the training fold only",
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "wall_min": wall / 60.0, "jobs": a.jobs,
        },
        "caps": caps,
        "curves": curves,
        "per_rung": per_rung,
        "sign_test": sign,
        "sparsity_slope": slopes,
        "records": recs,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[sparsity] model above permutation at "
          f"{sign['n_model_above']} of {sign['n_matched_rungs']} matched rungs; "
          f"median matched delta {sign['median_matched_delta']:+.4f}", flush=True)
    for at, sl in slopes.items():
        print(f"[sparsity] {at:12s} dAUC/dsensor = {sl['auc_per_sensor']:+.5f} "
              f"over n in {sl['n_range']}", flush=True)
    print(f"[sparsity] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
