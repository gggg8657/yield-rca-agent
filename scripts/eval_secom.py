"""Headline SECOM evaluation: baselines vs the agent loop, identical folds.

    OMP_NUM_THREADS=1 python scripts/eval_secom.py --repeats 5 --jobs 16

Writes ``runs/secom_eval.json``: every arm's per-fold AUC and average
precision, mean with a Student-t 95% CI over folds, and every pairwise
**paired** delta with a Wilcoxon p-value. Also runs the chronological split
(train on the earliest 70% of wafers, test on the last 30%) as a
deployment-realistic check that nothing depends on shuffling 90 days of fab
history.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from arms import AGENT_CFG, INNER, SEED, main_arms
from yieldrca.data import load_secom
from yieldrca.evaluate import (
    chronological_split,
    mean_ci,
    repeated_cv,
    rolling_origin_splits,
    summarize,
)


def _fit_score(arm, X, y, tr, te, **extra):
    est = clone(arm.factory()) if callable(arm.factory) else clone(arm.factory)
    est.fit(X[tr], y[tr])
    s = est.predict_proba(X[te])
    s = s[:, 1] if np.ndim(s) == 2 else s
    rec = {"arm": arm.name, "auc": float(roc_auc_score(y[te], s)),
           "ap": float(average_precision_score(y[te], s)),
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_fail_train": int(y[tr].sum()), "n_fail_test": int(y[te].sum()),
           **extra}
    if arm.extras is not None:
        rec.update(arm.extras(est))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=5,
                    help="contiguous time blocks for the rolling-origin protocol")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/secom_eval.json")
    a = ap.parse_args()

    X, y, names, t = load_secom(a.root, with_time=True)
    arms = main_arms()
    t0 = time.time()
    recs, folds = repeated_cv(arms, X, y, n_splits=a.splits,
                              n_repeats=a.repeats, seed=SEED, n_jobs=a.jobs)
    cv_s = time.time() - t0
    print(f"[cv] {len(recs)} fits in {cv_s/60:.1f} min")

    tr, te = chronological_split(t, 0.7)
    t1 = time.time()
    chrono = Parallel(n_jobs=min(a.jobs, len(arms)))(
        delayed(_fit_score)(arm, X, y, tr, te) for arm in arms)
    print(f"[chrono] {len(chrono)} fits in {(time.time()-t1)/60:.1f} min")

    roll = rolling_origin_splits(t, a.blocks)
    t2 = time.time()
    roll_recs = Parallel(n_jobs=a.jobs)(
        delayed(_fit_score)(arm, X, y, rtr, rte, origin=k)
        for arm in arms for k, (rtr, rte) in enumerate(roll))
    roll_recs = list(roll_recs)
    print(f"[rolling] {len(roll_recs)} fits in {(time.time()-t2)/60:.1f} min")
    rolling = {}
    for arm in arms:
        rs = [r for r in roll_recs if r["arm"] == arm.name]
        rs.sort(key=lambda r: r["origin"])
        rolling[arm.name] = {
            "per_origin": [{"origin": r["origin"], "auc": r["auc"],
                            "ap": r["ap"], "n_train": r["n_train"],
                            "n_test": r["n_test"],
                            "n_fail_test": r["n_fail_test"]} for r in rs],
            "summary": mean_ci([r["auc"] for r in rs]),
        }

    out = {
        "protocol": {
            "cv": f"RepeatedStratifiedKFold({a.splits} splits x {a.repeats} "
                  f"repeats, seed {SEED}) = {a.splits*a.repeats} folds",
            "inner_tuning": f"StratifiedKFold({INNER}) GridSearchCV on each "
                            "outer training fold, scoring=roc_auc",
            "leakage_control": "cleaning, imputation, scaling, screening, "
                               "attribution and verification are all fitted "
                               "inside the training fold only",
            "ci": "Student-t 95% CI over folds (df = n_folds - 1)",
            "paired": "per-fold difference, same train/test indices for every arm",
            "chronological": {
                "rule": "train on the earliest 70% of wafers by timestamp, "
                        "test on the last 30%",
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "n_fail_train": int(y[tr].sum()), "n_fail_test": int(y[te].sum()),
            },
            "rolling_origin": {
                "rule": f"{a.blocks} contiguous time blocks; train on blocks "
                        f"0..k, test on block k+1, for k = 0..{a.blocks - 2}",
                "n_origins": len(roll),
            },
            "agent_cfg": AGENT_CFG,
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cv_wall_min": cv_s / 60.0,
        },
        "arms": {a_.name: {"label": a_.label, "kind": a_.kind} for a_ in arms},
        "records": recs,
        "auc": summarize(recs, "auc"),
        "ap": summarize(recs, "ap"),
        "chronological": {r["arm"]: r for r in chrono},
        "rolling_origin": rolling,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'arm':<20} {'AUC':>6}  {'95% CI':>16}   {'AP':>6}   chrono AUC")
    for name, v in sorted(out["auc"]["per_arm"].items(), key=lambda kv: -kv[1]["mean"]):
        ci = f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
        print(f"{name:<20} {v['mean']:.3f}  {ci:>16}   "
              f"{out['ap']['per_arm'][name]['mean']:.3f}   "
              f"{out['chronological'][name]['auc']:.3f}")
    print("\nrolling-origin AUC (train on the past, test on the next block):")
    for name, v in out["rolling_origin"].items():
        per = " ".join(f"{r['auc']:.3f}" for r in v["per_origin"])
        print(f"  {name:<20} mean {v['summary']['mean']:.3f}   [{per}]")
    print("\npaired deltas vs rf_all (best baseline):")
    for k, d in out["auc"]["paired"].items():
        if k.endswith("__vs__rf_all"):
            print(f"  {k.split('__')[0]:<20} {d['mean']:+.4f} "
                  f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
                  f"W p={d.get('wilcoxon_p', float('nan')):.2g}  "
                  f"{d['wins']}W/{d['losses']}L")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
