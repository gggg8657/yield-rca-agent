"""Does the agent loop invent root causes when there are none?

The repo's pitch is that suspects failing a bootstrap stability check get
dropped, so the report is trustworthy. That is a rhetorical claim until someone
builds a world with no root causes and counts how many the loop reports anyway.

**The null.** Permute the labels. ``y`` keeps its 104 fails and 1,463 passes,
``X`` is untouched, and the assignment between them is destroyed -- so no sensor
carries information about failure, by construction. Anything the loop reports on
such a replicate is a false discovery: not "probably false", false.

**What is measured**, per replicate, for the pre-registered operating point:

* ``n_reported`` -- sensors handed to the engineer as root causes.
* ``n_merit`` -- how many cleared ``stability_min`` *on their own*, before
  ``AgentRCA.fit``'s never-return-empty-handed fallback tops the list back up.
  ``n_merit == 0`` and ``n_reported > 0`` is the fallback firing: the loop had
  nothing and said something.
* ``max_stability`` -- the largest bootstrap selection frequency any suspect
  reached. This is the statistic the drop step thresholds, so its null
  distribution is what tells us whether 0.3 is a decision or a formality.

**The comparison arm is the same loop on the real labels**, re-seeded. Both
sides then have a distribution over the same statistics and the question
"is the evidence for SECOM's suspects distinguishable from the evidence the
loop manufactures on noise?" becomes a two-sample question rather than an
opinion.

**The derived threshold.** tau(alpha) = the (1-alpha) quantile of the null
``max_stability``. A suspect scoring above tau is one the null produces less
than alpha of the time; requiring it is Westfall-Young max-statistic control,
which is family-wise over the sensors screened in that replicate. Counting how
many real-label suspects clear tau is the honest version of "how many root
causes does SECOM support".

    OMP_NUM_THREADS=1 python scripts/null_fdr.py --null 200 --real 40 --jobs 16
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

from arms import AGENT_CFG, SEED, agent
from yieldrca.data import load_secom


def _one(X, y, permuted: bool, rep: int, base: str, over=None):
    """One agent-loop fit; returns what it reported and how well supported it was.

    ``permuted`` shuffles the labels with a replicate-specific seed. The loop's
    own ``random_state`` also varies with ``rep`` so that the real-label arm is
    a distribution over the loop's internal randomness rather than one point --
    otherwise the two arms would not be comparable as samples.
    """
    rng = np.random.default_rng(10_000 + rep)
    yy = rng.permutation(y) if permuted else y
    est = agent(base, random_state=SEED + rep, **(over or {})).fit(X, yy)

    stab = {int(j): float(v) for j, v in est.stability_.items()}
    stab_top = {int(j): float(v) for j, v in est.stability_top_k_.items()}
    thr = est.stability_min
    n_merit = int(sum(v >= thr for v in stab.values()))
    return {
        "rep": rep,
        "permuted": bool(permuted),
        "base": base,
        "overrides": dict(over or {}),
        "n_fail": int(yy.sum()),
        "n_reported": int(len(est.selected_)),
        "n_candidates": int(est.n_candidates_),
        "n_merit": n_merit,
        "fallback_fired": bool(n_merit == 0),
        "stability_min": float(thr),
        "max_stability": float(max(stab.values())) if stab else 0.0,
        "max_stability_top5": float(max(stab_top.values())) if stab_top else 0.0,
        "stability_values": sorted(stab.values(), reverse=True),
        "top5": [int(j) for j in est.ranking()[:5]],
        "selected": [int(j) for j in est.selected_original_],
    }


def _q(a, p):
    return float(np.quantile(np.asarray(a, dtype=float), p)) if len(a) else float("nan")


def _summ(recs):
    """Aggregate one arm's replicates. Every field here is a count or a quantile
    over ``recs``; nothing is asserted that a replicate did not produce."""
    rep = [r["n_reported"] for r in recs]
    mer = [r["n_merit"] for r in recs]
    mx = [r["max_stability"] for r in recs]
    return {
        "n_replicates": len(recs),
        "n_reported_mean": float(np.mean(rep)) if rep else float("nan"),
        "n_reported_median": float(np.median(rep)) if rep else float("nan"),
        "n_reported_min": int(min(rep)) if rep else 0,
        "n_reported_max": int(max(rep)) if rep else 0,
        "abstention_rate": float(np.mean([r == 0 for r in rep])) if rep else float("nan"),
        "n_merit_mean": float(np.mean(mer)) if mer else float("nan"),
        "merit_zero_rate": float(np.mean([m == 0 for m in mer])) if mer else float("nan"),
        "fallback_rate": float(np.mean([r["fallback_fired"] for r in recs])) if recs else float("nan"),
        "max_stability_mean": float(np.mean(mx)) if mx else float("nan"),
        "max_stability_q05": _q(mx, 0.05),
        "max_stability_median": _q(mx, 0.50),
        "max_stability_q95": _q(mx, 0.95),
        "total_reported": int(sum(rep)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--real", type=int, default=40)
    ap.add_argument("--base", default="rf")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--root", default="data")
    ap.add_argument("--select-k", type=int, default=None,
                    help="override the loop's bootstrap selection depth. H4: "
                         "the univariate arm's error control jumped when its "
                         "depth narrowed, so this asks whether depth or the "
                         "permutation-importance estimator is what limits the "
                         "loop's.")
    ap.add_argument("--out", default="runs/null_fdr.json")
    a = ap.parse_args()

    X, y, names = load_secom(a.root)
    over = {"select_k": a.select_k} if a.select_k is not None else {}
    jobs = [(True, r) for r in range(a.null)] + [(False, r) for r in range(a.real)]
    if over:
        print(f"[null_fdr] overriding {over} (baseline is "
              f"select_k={AGENT_CFG['select_k']})", flush=True)
    print(f"[null_fdr] {len(jobs)} agent-loop fits "
          f"({a.null} permuted, {a.real} real), {a.jobs} workers", flush=True)

    t0 = time.time()
    recs = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(_one)(X, y, perm, rep, a.base, over) for perm, rep in jobs)
    wall = time.time() - t0
    print(f"[null_fdr] done in {wall/60:.1f} min", flush=True)

    null = [r for r in recs if r["permuted"]]
    real = [r for r in recs if not r["permuted"]]
    null_max = [r["max_stability"] for r in null]

    # Westfall-Young style max-statistic thresholds from the null.
    taus = {f"alpha_{al}": _q(null_max, 1 - al) for al in (0.10, 0.05, 0.01)}

    # How many real-label suspects clear each threshold?
    cleared = {}
    for key, tau in taus.items():
        per = [int(sum(v >= tau for v in r["stability_values"])) for r in real]
        cleared[key] = {
            "tau": tau,
            "n_cleared_mean": float(np.mean(per)) if per else float("nan"),
            "n_cleared_median": float(np.median(per)) if per else float("nan"),
            "n_cleared_min": int(min(per)) if per else 0,
            "n_cleared_max": int(max(per)) if per else 0,
            "replicates_with_none": float(np.mean([p == 0 for p in per])) if per else float("nan"),
        }

    # Two-sample separation of the statistic the drop step thresholds.
    real_max = [r["max_stability"] for r in real]
    from scipy.stats import mannwhitneyu
    u_stat = p_val = float("nan")
    if null_max and real_max:
        u_stat, p_val = mannwhitneyu(real_max, null_max, alternative="greater")
    # P(real replicate's max > null replicate's max), the common-language effect
    # size; 0.5 means the loop's confidence carries no information about whether
    # the labels were real.
    auc_sep = float(np.mean([[1.0 if rr > nn else 0.5 if rr == nn else 0.0
                              for nn in null_max] for rr in real_max])) \
        if (null_max and real_max) else float("nan")

    out = {
        "protocol": {
            "null": "labels permuted over all wafers (class balance preserved "
                    "exactly: 104 fails / 1,463 passes); X untouched. Every "
                    "sensor reported under this null is a false discovery by "
                    "construction.",
            "real_arm": "identical loop, true labels, random_state varied per "
                        "replicate so both arms are distributions over the "
                        "loop's internal randomness",
            "fit_on": "all 1,567 wafers (this measures what the loop *reports*, "
                      "not what it predicts; no held-out scoring is involved)",
            "n_null": a.null, "n_real": a.real, "base": a.base,
            "overrides": over,
            "threshold": "tau(alpha) = (1-alpha) quantile of the null "
                         "max_stability; Westfall-Young max-statistic control, "
                         "family-wise over the sensors screened per replicate",
            "agent_cfg": {**AGENT_CFG, **over},
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "wall_min": wall / 60.0,
        },
        "null": _summ(null),
        "real": _summ(real),
        "null_fdr": {
            "definition": "every sensor reported on a permuted-label replicate "
                          "is false, so FDR = (reported on null) / (reported on "
                          "null) = 1.0 whenever anything is reported at all. "
                          "The informative quantity is the abstention rate.",
            "false_discoveries_total": int(sum(r["n_reported"] for r in null)),
            "fdr_given_nonempty": 1.0 if sum(r["n_reported"] for r in null) else 0.0,
            "abstention_rate": _summ(null)["abstention_rate"],
        },
        "thresholds": taus,
        "real_cleared": cleared,
        "separation": {
            "mannwhitney_u": float(u_stat),
            "p_real_greater": float(p_val),
            "prob_real_max_exceeds_null_max": auc_sep,
        },
        "records": recs,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    n, rl = out["null"], out["real"]
    print(f"\n{'':<28}{'null (permuted)':>18}{'real labels':>16}")
    for lab, key in [("sensors reported (mean)", "n_reported_mean"),
                     ("cleared threshold on merit", "n_merit_mean"),
                     ("abstention rate", "abstention_rate"),
                     ("fallback fired", "fallback_rate"),
                     ("max bootstrap support", "max_stability_mean")]:
        print(f"{lab:<28}{n[key]:>18.3f}{rl[key]:>16.3f}")
    print(f"\nfalse discoveries on the null: {out['null_fdr']['false_discoveries_total']} "
          f"sensors over {a.null} replicates")
    print(f"P(real max > null max) = {auc_sep:.3f}  (0.5 = no information)  "
          f"Mann-Whitney p = {p_val:.3g}")
    for key, c in cleared.items():
        print(f"  tau {key:<12} = {c['tau']:.3f} -> real suspects clearing it: "
              f"mean {c['n_cleared_mean']:.2f}, "
              f"{c['replicates_with_none']:.0%} of replicates clear none")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
