"""H7: does the attribution statistic close the agent loop's AUC deficit?

The loop scores 0.717 [0.699, 0.735] against `rf_all`'s 0.759, a paired
-0.042 [-0.058, -0.025] over 25 folds. Two of this repository's measurements
now show its attribution statistic is its weakest component -- swapping it is
worth +13.0 points of top-5 stability (H5) and +2.2 points of null error
control (H6) -- so the obvious question is whether it also explains the AUC
gap.

**One arm, added under the published protocol rather than a re-run of it.**
`RepeatedStratifiedKFold(5 x 5, seed 0)` is deterministic, so the folds here
are byte-identical to `runs/secom_eval.json`'s and the paired deltas are
computed against that file's stored per-fold AUCs. Nothing already measured is
recomputed, so nothing already published can move.

    OMP_NUM_THREADS=1 python scripts/eval_attr_arm.py --jobs 16
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


def _agent_model(base="rf"):
    """The pre-registered loop with `attribution="model"` and nothing else."""
    cfg = dict(AGENT_CFG)
    cfg["attribution"] = "model"
    return AgentRCA(base=base, base_kw=AGENT_BASE_KW, **cfg)


def _one(X, y, tr, te, fold, base):
    est = clone(_agent_model(base)).fit(X[tr], y[tr])
    p = est.predict_proba(X[te])[:, 1]
    return {
        "fold": fold,
        "auc": float(roc_auc_score(y[te], p)),
        "ap": float(average_precision_score(y[te], p)),
        "n_selected": int(len(est.selected_)),
        "n_candidates": int(est.n_candidates_),
        "selected": est.selected_original_.tolist(),
        "top5": [int(j) for j in est.ranking()[:5]],
        "n_test": int(len(te)), "n_fail_test": int(y[te].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--base", default="rf")
    ap.add_argument("--root", default="data")
    ap.add_argument("--eval", default="runs/secom_eval.json")
    ap.add_argument("--out", default="runs/attr_arm.json")
    a = ap.parse_args()

    X, y, _ = load_secom(a.root)
    cv = RepeatedStratifiedKFold(n_splits=a.splits, n_repeats=a.repeats,
                                 random_state=SEED)
    folds = list(cv.split(X, y))
    name = f"agent_model_{a.base}"
    print(f"[attr_arm] {name}: {len(folds)} folds, {a.jobs} workers", flush=True)

    t0 = time.time()
    recs = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(_one)(X, y, tr, te, i, a.base)
        for i, (tr, te) in enumerate(folds))
    wall = time.time() - t0
    recs.sort(key=lambda r: r["fold"])
    auc = mean_ci([r["auc"] for r in recs])
    print(f"[attr_arm] {name} AUC {auc['mean']:.3f} "
          f"[{auc['ci_lo']:.3f}, {auc['ci_hi']:.3f}]  "
          f"({wall/60:.1f} min)", flush=True)

    # Paired against the stored per-fold AUCs of the published arms. Same folds
    # by construction, and asserted rather than assumed.
    ev = json.loads(Path(a.eval).read_text())
    mine = {r["fold"]: r["auc"] for r in recs}
    paired, ref = {}, {}
    for arm in sorted({r["arm"] for r in ev["records"]}):
        theirs = {r["fold"]: r["auc"] for r in ev["records"]
                  if r["arm"] == arm}
        # Guard the pairing hard. A subset of the reference arm's folds would
        # pair happily and mean nothing -- the smoke run of this script did
        # exactly that with --splits 2, comparing 2 of the reference's 25
        # folds and printing a confident interval. Require set equality, so a
        # reduced-fold invocation refuses to emit a paired delta at all.
        if set(mine) != set(theirs):
            print(f"  ! {arm}: fold sets differ "
                  f"({len(mine)} vs {len(theirs)}), refusing to pair",
                  flush=True)
            continue
        common = sorted(mine)
        paired[f"{name}__vs__{arm}"] = paired_delta(
            [mine[f] for f in common], [theirs[f] for f in common])
        ref[arm] = mean_ci([theirs[f] for f in common])
        d = paired[f"{name}__vs__{arm}"]
        print(f"  vs {arm:16s} {d['mean']:+.4f} "
              f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  "
              f"({d['wins']}/{d['losses']} folds)", flush=True)

    out = {
        "protocol": {
            "hypothesis": "H7. The loop's AUC deficit against rf_all is the "
                          "price of sparsity rather than of ranking quality, "
                          "so swapping the attribution statistic moves the "
                          "number only as far as a good ranker at the same "
                          "budget already does (univar_top25_rf, -0.029) and "
                          "leaves a negative paired delta whose CI excludes "
                          "zero.",
            "arm": name,
            "change_from_pre_registered": {"attribution": "model"},
            "cv": f"RepeatedStratifiedKFold({a.splits} splits x {a.repeats} "
                  f"repeats, seed {SEED}) = {len(folds)} folds -- "
                  f"byte-identical to {a.eval}, which is why the deltas below "
                  f"are paired against that file's stored per-fold AUCs "
                  f"instead of re-running its arms",
            "leakage_control": "cleaning, screening, attribution and "
                               "verification all fitted inside the training "
                               "fold; the loop is an sklearn estimator and "
                               "never sees the test rows",
            "reference_file": a.eval,
            "reference_cv": ev["protocol"]["cv"],
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "wall_min": wall / 60.0, "jobs": a.jobs,
        },
        "arm": name,
        "auc": auc,
        "ap": mean_ci([r["ap"] for r in recs]),
        "n_selected_mean": float(np.mean([r["n_selected"] for r in recs])),
        "records": recs,
        "paired": paired,
        "reference_arms": ref,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[attr_arm] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
