"""Is the agent loop's bootstrap support better calibrated than a plain ranker's?

`runs/null_fdr.json` shows the agent loop reports root causes on permuted
labels, and that a threshold calibrated on that null restores control because
the loop's support statistic separates the two worlds at P(real > null) = 0.873.
This script asks the question that number invites: **would anything simpler have
separated them just as well?**

Comparing raw false-discovery rates would be pointless -- any procedure that
always emits a top-5 has FDR 1.0 under this null, univariate ranking included.
The informative quantity is the *separation* the statistic achieves, because
that is what decides how much of a report survives calibration.

**The arms.** Each computes a bootstrap selection frequency per sensor, then the
maximum over sensors, exactly as `AgentRCA`'s VerifierAgent does:

* ``univariate``  -- rank sensors by |per-sensor rank AUC - 0.5|.
* ``logreg_coef`` -- rank by |standardised L2 logistic coefficient| (multivariate,
  but one fit, no permutation and no verification loop).
* ``rf_impurity`` -- rank by random-forest impurity decrease.

**The confound this controls.** Separation grows with how many bootstrap draws
go into the statistic, so every arm here uses the agent loop's own ``n_boot``
and ``select_k`` from ``AGENT_CFG`` rather than a convenient value, and is
scored with the identical ``max`` statistic. A difference in separation is then
a difference in the ranker, which is the thing being compared.

Cleaning is fitted inside each bootstrap resample, as everywhere else in this
repo, so no arm sees a decision made on rows it is scored against.

    OMP_NUM_THREADS=1 python scripts/null_fdr_rankers.py --null 200 --real 40 --jobs 16
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
from scipy.stats import mannwhitneyu

from arms import AGENT_CFG, SEED
from yieldrca.attribution import screen_model, screen_univariate, screen_multivariate
from yieldrca.data import load_secom
from yieldrca.estimator import make_rf
from yieldrca.preprocess import SensorCleaner

RANKERS = ("univariate", "logreg_coef", "rf_impurity")


def _rank(kind, X, y, seed):
    """Ordered sensor indices for one resample, cleaning fitted in-sample."""
    if kind == "univariate":
        _, w = screen_univariate(X, y, n_keep=1)
    elif kind == "logreg_coef":
        _, w = screen_multivariate(X, y, n_keep=1, seed=seed)
    elif kind == "rf_impurity":
        _, w = screen_model(
            lambda a, b: make_rf(n_estimators=300, seed=seed).fit(a, b),
            X, y, n_keep=1)
    else:
        raise ValueError(kind)
    return np.argsort(w)[::-1]


def _one(X, y, permuted, rep, kind, n_boot, select_k, top_k):
    """One replicate: bootstrap selection frequencies, then the max statistic."""
    rng = np.random.default_rng(10_000 + rep)
    yy = rng.permutation(y) if permuted else y

    cleaner = SensorCleaner().fit(X)
    Xc = cleaner.transform(X)
    keep = cleaner.keep_

    counts = np.zeros(Xc.shape[1])
    n_eff = 0
    boot = np.random.default_rng(SEED + 100 + rep)
    for b in range(n_boot):
        idx = boot.integers(0, len(yy), len(yy))
        if yy[idx].sum() < 5:
            continue
        order = _rank(kind, Xc[idx], yy[idx], SEED + rep + b)
        counts[order[:select_k]] += 1
        n_eff += 1
    supp = counts / max(n_eff, 1)

    full = _rank(kind, Xc, yy, SEED + rep)
    top = [int(keep[j]) for j in full[:top_k]]
    return {
        "rep": rep, "permuted": bool(permuted), "ranker": kind,
        "max_stability": float(supp.max()),
        "n_at_or_above_half": int((supp >= 0.5).sum()),
        "stability_values": sorted(supp[supp > 0].tolist(), reverse=True)[:60],
        "top5": top,
    }


def _summ(recs):
    mx = [r["max_stability"] for r in recs]
    return {
        "n_replicates": len(recs),
        "max_stability_mean": float(np.mean(mx)) if mx else float("nan"),
        "max_stability_q05": float(np.quantile(mx, 0.05)) if mx else float("nan"),
        "max_stability_q95": float(np.quantile(mx, 0.95)) if mx else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--real", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--root", default="data")
    ap.add_argument("--agent", default="runs/null_fdr.json",
                    help="the agent-loop run this is compared against")
    ap.add_argument("--out", default="runs/null_fdr_rankers.json")
    a = ap.parse_args()

    X, y, names = load_secom(a.root)
    n_boot = AGENT_CFG["n_boot"]
    select_k = AGENT_CFG["select_k"]
    top_k = AGENT_CFG["top_k"]
    jobs = [(k, p, r) for k in RANKERS
            for p, n in ((True, a.null), (False, a.real))
            for r in range(n)]
    print(f"[rankers] {len(jobs)} replicates over {len(RANKERS)} rankers, "
          f"n_boot={n_boot} select_k={select_k} (from AGENT_CFG), "
          f"{a.jobs} workers", flush=True)

    t0 = time.time()
    recs = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(_one)(X, y, p, r, k, n_boot, select_k, top_k)
        for k, p, r in jobs)
    wall = time.time() - t0
    print(f"[rankers] done in {wall/60:.1f} min", flush=True)

    agent = None
    ap_ = Path(a.agent)
    if ap_.exists():
        agent = json.loads(ap_.read_text())

    per = {}
    for kind in RANKERS:
        nl = [r for r in recs if r["ranker"] == kind and r["permuted"]]
        rl = [r for r in recs if r["ranker"] == kind and not r["permuted"]]
        nm = [r["max_stability"] for r in nl]
        rm = [r["max_stability"] for r in rl]
        u, pv = mannwhitneyu(rm, nm, alternative="greater") if (nm and rm) \
            else (float("nan"), float("nan"))
        sep = float(np.mean([[1.0 if x > z else 0.5 if x == z else 0.0
                              for z in nm] for x in rm])) if (nm and rm) else float("nan")
        tau = float(np.quantile(nm, 0.95)) if nm else float("nan")
        kept = [int(sum(v >= tau for v in r["stability_values"])) for r in rl]
        per[kind] = {
            "null": _summ(nl), "real": _summ(rl),
            "prob_real_max_exceeds_null_max": sep,
            "mannwhitney_p": float(pv),
            "tau_alpha_0.05": tau,
            "real_reported_at_tau_mean": float(np.mean(kept)) if kept else float("nan"),
            "real_abstention_at_tau": float(np.mean([k == 0 for k in kept]))
            if kept else float("nan"),
        }

    if agent:
        nm = [r["max_stability"] for r in agent["records"] if r["permuted"]]
        rm = [r["max_stability"] for r in agent["records"] if not r["permuted"]]
        tau = float(np.quantile(nm, 0.95))
        kept = [int(sum(v >= tau for v in r["stability_values"]))
                for r in agent["records"] if not r["permuted"]]
        per["agent (full loop)"] = {
            "null": {"n_replicates": len(nm),
                     "max_stability_mean": float(np.mean(nm)),
                     "max_stability_q05": float(np.quantile(nm, 0.05)),
                     "max_stability_q95": float(np.quantile(nm, 0.95))},
            "real": {"n_replicates": len(rm),
                     "max_stability_mean": float(np.mean(rm)),
                     "max_stability_q05": float(np.quantile(rm, 0.05)),
                     "max_stability_q95": float(np.quantile(rm, 0.95))},
            "prob_real_max_exceeds_null_max":
                agent["separation"]["prob_real_max_exceeds_null_max"],
            "mannwhitney_p": agent["separation"]["p_real_greater"],
            "tau_alpha_0.05": tau,
            "real_reported_at_tau_mean": float(np.mean(kept)) if kept else float("nan"),
            "real_abstention_at_tau": float(np.mean([k == 0 for k in kept]))
            if kept else float("nan"),
            "source": a.agent,
        }

    best = max(per, key=lambda k: per[k]["prob_real_max_exceeds_null_max"])
    out = {
        "protocol": {
            "question": "does the agent loop's bootstrap support separate a "
                        "world with causes from one without any better than a "
                        "plain ranker's does?",
            "why_not_fdr": "any procedure that always emits a top-k has FDR 1.0 "
                           "under this null, so raw FDR cannot distinguish the "
                           "arms; separation decides how much of a report "
                           "survives calibration, so separation is compared",
            "null": "labels permuted over all wafers, class balance preserved; "
                    "X untouched",
            "matched": f"every arm uses the agent loop's own n_boot={n_boot} "
                       f"and select_k={select_k} from AGENT_CFG, and the "
                       f"identical max-over-sensors statistic, so a difference "
                       f"in separation is a difference in the ranker",
            "leakage_control": "SensorCleaner fitted inside each resample",
            "n_null": a.null, "n_real": a.real, "top_k": top_k,
            "agent_arm_source": a.agent if agent else None,
        },
        "environment": {"python": platform.python_version(),
                        "sklearn": sklearn.__version__,
                        "numpy": np.__version__,
                        "platform": platform.platform(),
                        "wall_min": wall / 60.0},
        "best_separating_arm": best,
        "per_ranker": per,
        "records": recs,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'arm':<22}{'null max':>10}{'real max':>10}{'P(real>null)':>14}"
          f"{'tau .05':>9}{'suspects':>10}{'abstains':>10}")
    for kind, v in sorted(per.items(),
                          key=lambda kv: -kv[1]["prob_real_max_exceeds_null_max"]):
        print(f"{kind:<22}{v['null']['max_stability_mean']:>10.3f}"
              f"{v['real']['max_stability_mean']:>10.3f}"
              f"{v['prob_real_max_exceeds_null_max']:>14.3f}"
              f"{v['tau_alpha_0.05']:>9.3f}"
              f"{v['real_reported_at_tau_mean']:>10.2f}"
              f"{v['real_abstention_at_tau']:>10.0%}")
    print(f"\nbest separating arm: {best}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
