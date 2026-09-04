"""The only benchmark in this repo with ground-truth root causes.

    OMP_NUM_THREADS=1 python scripts/eval_synthetic.py --seeds 10 --jobs 16

SECOM has no ground-truth causal sensors, so no recovery claim can be made on
it -- only prediction and selection stability. This script is where recovery
*is* measurable: the generator plants ``n_causal`` sensors that actually drive
the label among ``p`` correlated decoys, so precision and recall of the
recovered set are defined.

Reported per method, averaged over ``--seeds`` independently generated
datasets:

* **top-5 recall / precision** against the planted set,
* **selected-set recall / precision** for the methods that select,
* **held-out AUC** under the same repeated stratified CV protocol as SECOM,
* **top-5 stability** under the same definition as SECOM
  (``yieldrca/stability.py``), so the KPI is exercised where truth is known.

Writes ``runs/synthetic.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from arms import AGENT_BASE_KW, AGENT_CFG, SEED, rf_all, univar_rf
from yieldrca.data import make_synthetic
from yieldrca.estimator import AgentRCA
from yieldrca.evaluate import mean_ci, paired_delta
from yieldrca.preprocess import SensorCleaner
from yieldrca.stability import bootstrap_replicates, measure

SYN_CFG = dict(AGENT_CFG)
SYN_CFG.update(n_screen=60, n_screen_boot=40, select_k=20, stability_min=0.3,
               max_select=15)


def _rank_univariate(X, y):
    from yieldrca.attribution import screen_univariate

    cl = SensorCleaner().fit(X)
    _, w = screen_univariate(cl.transform(X), y, n_keep=1)
    return cl.keep_[np.argsort(w)[::-1]]


def _rank_rf(X, y):
    from yieldrca.attribution import screen_model
    from yieldrca.estimator import make_rf

    cl = SensorCleaner().fit(X)
    _, w = screen_model(lambda a, b: make_rf(n_estimators=300).fit(a, b),
                        cl.transform(X), y, n_keep=1)
    return cl.keep_[np.argsort(w)[::-1]]


def _rank_agent(X, y):
    """The loop's reported ranking -- survivors first (see AgentRCA.ranking)."""
    return AgentRCA(base="rf", base_kw=AGENT_BASE_KW, **SYN_CFG) \
        .fit(X, y).ranking()


def _one_seed(seed, k, cv_splits):
    X, y, names, causal = make_synthetic(seed=seed)
    truth = set(int(c) for c in causal)
    rec = {"seed": seed, "n_causal": len(truth), "fail_rate": float(y.mean()),
           "methods": {}}

    for name, fn in (("univariate", _rank_univariate), ("rf_impurity", _rank_rf),
                     ("agent", _rank_agent)):
        order = fn(X, y)
        top = set(int(j) for j in order[:k])
        rec["methods"][name] = {
            "topk_hits": len(top & truth),
            "topk_recall": len(top & truth) / len(truth),
            "topk_precision": len(top & truth) / k,
            "topk": sorted(top),
        }

    m = AgentRCA(base="rf", base_kw=AGENT_BASE_KW, **SYN_CFG).fit(X, y)
    sel = set(int(j) for j in m.selected_original_)
    rec["methods"]["agent"].update({
        "n_selected": len(sel),
        "selected_recall": len(sel & truth) / len(truth),
        "selected_precision": len(sel & truth) / max(len(sel), 1),
        "selected": sorted(sel),
    })

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    for name, factory in (("rf_all", rf_all), ("univar_top25_rf", univar_rf),
                          ("agent_rf", lambda: AgentRCA(base="rf",
                                                        base_kw=AGENT_BASE_KW,
                                                        **SYN_CFG))):
        aucs = []
        for tr, te in cv.split(X, y):
            est = factory().fit(X[tr], y[tr])
            s = est.predict_proba(X[te])
            aucs.append(float(roc_auc_score(y[te], s[:, 1] if np.ndim(s) == 2 else s)))
        rec.setdefault("auc", {})[name] = aucs
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default="runs/synthetic.json")
    a = ap.parse_args()

    t0 = time.time()
    recs = Parallel(n_jobs=min(a.jobs, a.seeds), verbose=5)(
        delayed(_one_seed)(s, a.k, a.cv_splits) for s in range(a.seeds))
    recs = list(recs)

    methods = list(recs[0]["methods"])
    summary = {}
    for m in methods:
        summary[m] = {
            key: mean_ci([r["methods"][m][key] for r in recs])
            for key in ("topk_recall", "topk_precision", "topk_hits")
        }
        if "selected_recall" in recs[0]["methods"][m]:
            for key in ("selected_recall", "selected_precision", "n_selected"):
                summary[m][key] = mean_ci([r["methods"][m][key] for r in recs])

    auc_arms = list(recs[0]["auc"])
    per_fold = {arm: [v for r in recs for v in r["auc"][arm]] for arm in auc_arms}
    auc_summary = {arm: mean_ci(v) for arm, v in per_fold.items()}
    paired = {f"{arm}__vs__rf_all": paired_delta(per_fold[arm], per_fold["rf_all"])
              for arm in auc_arms if arm != "rf_all"}

    # top-5 stability on one synthetic dataset, same definition as SECOM
    X, y, names, causal = make_synthetic(seed=0)
    reps = bootstrap_replicates(y, n_boot=a.boot, seed=SEED)
    stab = {}
    for name, fn in (("univariate", _rank_univariate), ("rf_impurity", _rank_rf),
                     ("agent", _rank_agent)):
        r = measure(fn, X, y, reps, k=a.k, cmap=None, n_jobs=a.jobs)
        r.pop("top_sets", None)
        stab[name] = r

    out = {
        "generator": {"n": 1500, "p": 200, "n_causal": 5, "fail_rate_target": 0.07,
                      "missing_rate": 0.04,
                      "note": "block-correlated noise (blocks of 20) so raw "
                              "correlation alone cannot find the causal set"},
        "protocol": {
            "seeds": a.seeds,
            "k": a.k,
            "recovery": f"top-{a.k} vs the planted causal set, on the full dataset",
            "auc": f"StratifiedKFold({a.cv_splits}) per seed, "
                   f"{a.seeds * a.cv_splits} folds pooled",
            "stability": f"{a.boot} bootstrap resamples, seed-0 dataset, "
                         "definition in yieldrca/stability.py",
            "agent_cfg": SYN_CFG,
        },
        "wall_min": (time.time() - t0) / 60.0,
        "recovery": summary,
        "auc": {"per_arm": auc_summary, "paired": paired},
        "stability": stab,
        "records": recs,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'method':<14} top5 recall  top5 prec   stability(pairwise)")
    for m in methods:
        print(f"{m:<14} {summary[m]['topk_recall']['mean']:.2f}         "
              f"{summary[m]['topk_precision']['mean']:.2f}        "
              f"{stab[m]['raw']['pairwise_overlap']:.3f}")
    print(f"\n{'arm':<18} CV AUC")
    for arm, v in auc_summary.items():
        print(f"{arm:<18} {v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    print(f"\nwrote {a.out} ({out['wall_min']:.1f} min)")


if __name__ == "__main__":
    main()
