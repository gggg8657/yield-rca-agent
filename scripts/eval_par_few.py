"""What was the `PredictAllReportFew` predictor mismatch worth?

`PredictAllReportFew` is the configuration this repository's recommendations
argue for: predict with every sensor, report with the agent loop. Its docstring
claimed its held-out AUC was the baseline row's "by construction", on the
grounds that the loop never touches ``predict_proba``. That argument is valid
for the loop and was then extended one step too far: the default predictor was
a forest with ``min_samples_leaf`` fixed at 5, while `rf_all` -- the arm the
0.759 baseline row comes from -- tunes it over {1, 5, 10} by inner CV. An
untuned forest is not a tuned one, so the equality was asserted rather than
measured.

This measures it. Two predictors under the **same 25 folds** as
`runs/secom_eval.json` (RepeatedStratifiedKFold 5x5, seed 0), paired per fold:

* ``par_untuned`` -- the old default, ``min_samples_leaf=5`` fixed.
* ``par_tuned``   -- the new default, :func:`make_rf_tuned`.

Only the predictors are evaluated, not the whole class. That is deliberate and
it is not a shortcut: ``predict_proba`` provably reads ``predictor_`` alone, so
fitting the agent loop 25 more times could not move an AUC. The structural half
of the claim is pinned by a test instead
(`test_predict_all_report_few_predicts_with_every_sensor`).

Neither arm is byte-identical to ``rf_all``, which wraps ``SensorCleaner``
inside its grid search while this class cleans once in ``fit``; the difference
can only reach hyperparameter choice, not the held-out estimate, and is
recorded in the JSON rather than papered over.

    OMP_NUM_THREADS=1 python scripts/eval_par_few.py --jobs 1
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
from sklearn.model_selection import RepeatedStratifiedKFold

from yieldrca.data import load_secom
from yieldrca.estimator import make_rf, make_rf_tuned
from yieldrca.evaluate import mean_ci, paired_delta
from yieldrca.preprocess import SensorCleaner

SEED = 0


def _arms():
    """The old default and the new one. Cleaning happens outside, as in `fit`."""
    return {
        "par_untuned": ("old default: median-impute -> RF, "
                        "min_samples_leaf=5 fixed", make_rf),
        "par_tuned": ("new default: median-impute -> RF, min_samples_leaf "
                      "tuned over {1, 5, 10} by inner CV", make_rf_tuned),
    }


def _one(X, y, tr, te, ctor, fold):
    """One fold of one arm. Cleaning is fitted on the training rows only."""
    cl = SensorCleaner().fit(X[tr])
    est = clone(ctor()).fit(cl.transform(X[tr]), y[tr])
    p = est.predict_proba(cl.transform(X[te]))[:, 1]
    out = {"fold": fold, "auc": float(roc_auc_score(y[te], p)),
           "ap": float(average_precision_score(y[te], p))}
    best = getattr(est, "best_params_", None)
    if best:
        out["best_params"] = {k: v for k, v in best.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1,
                    help="kept at 1 by default: this runs alongside other "
                         "jobs holding the 16-worker lease")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/par_few.json")
    a = ap.parse_args()

    X, y, _ = load_secom(a.root)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    folds = list(cv.split(X, y))
    arms = _arms()
    print(f"[par_few] {len(arms)} arms x {len(folds)} folds, "
          f"{a.jobs} worker(s)", flush=True)

    t0 = time.time()
    per_arm = {}
    for name, (label, ctor) in arms.items():
        recs = Parallel(n_jobs=a.jobs, verbose=1)(
            delayed(_one)(X, y, tr, te, ctor, i)
            for i, (tr, te) in enumerate(folds))
        per_arm[name] = {"label": label, "records": recs,
                         "auc": mean_ci([r["auc"] for r in recs]),
                         "ap": mean_ci([r["ap"] for r in recs])}
        print(f"  {name:14s} AUC {per_arm[name]['auc']['mean']:.3f} "
              f"[{per_arm[name]['auc']['ci_lo']:.3f}, "
              f"{per_arm[name]['auc']['ci_hi']:.3f}]", flush=True)
    wall = time.time() - t0

    a_auc = [r["auc"] for r in per_arm["par_tuned"]["records"]]
    b_auc = [r["auc"] for r in per_arm["par_untuned"]["records"]]
    delta = paired_delta(a_auc, b_auc)

    # How often did tuning actually pick something other than the old fixed 5?
    chosen = [r.get("best_params", {}).get("clf__min_samples_leaf")
              for r in per_arm["par_tuned"]["records"]]
    chosen = [c for c in chosen if c is not None]
    picks = {str(v): int(sum(c == v for c in chosen)) for v in sorted(set(chosen))}

    out = {
        "protocol": {
            "question": "PredictAllReportFew's docstring claimed its held-out "
                        "AUC was the baseline row's by construction. The loop "
                        "not touching predict_proba makes the loop free; it "
                        "does not make an untuned forest equal a tuned one. "
                        "This measures the difference the claim skipped over.",
            "cv": "RepeatedStratifiedKFold(5 splits x 5 repeats, seed 0) = "
                  "25 folds, the same protocol as runs/secom_eval.json",
            "paired": "per-fold difference, identical train/test indices for "
                      "both arms",
            "leakage_control": "SensorCleaner and the imputer are fitted on "
                               "the training rows of each fold only; the "
                               "inner GridSearchCV runs inside the training "
                               "fold",
            "not_identical_to_rf_all": "rf_all wraps SensorCleaner inside its "
                                       "grid search, while PredictAllReportFew "
                                       "cleans once in fit. Cleaning is "
                                       "unsupervised and training-fold-only in "
                                       "both, so this can reach hyperparameter "
                                       "choice but not the held-out estimate. "
                                       "These rows are therefore comparable to "
                                       "each other exactly, and to the rf_all "
                                       "row closely but not byte-identically.",
            "predict_proba_ignores_the_loop": "asserted by "
                "tests/test_real.py::test_predict_all_report_few_predicts_"
                "with_every_sensor, not re-measured here",
            "n_folds": len(folds), "seed": SEED,
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "wall_min": wall / 60.0,
            "jobs": a.jobs,
        },
        "per_arm": per_arm,
        "paired_tuned_minus_untuned": delta,
        "tuned_min_samples_leaf_picks": picks,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[par_few] tuned - untuned = {delta['mean']:+.4f} "
          f"[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}] "
          f"({delta['wins']} folds better, {delta['losses']} worse)", flush=True)
    print(f"[par_few] inner CV picked min_samples_leaf: {picks}", flush=True)
    print(f"[par_few] done in {wall/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
