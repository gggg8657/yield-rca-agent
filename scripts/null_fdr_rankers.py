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

**Where the cleaning sits.** ``SensorCleaner`` is fitted once per replicate on
the full matrix, *outside* the bootstrap loop -- not inside each resample. That
is deliberate and it is stated here because an earlier version of this docstring
claimed the opposite and was wrong. The cleaner is unsupervised: it drops
all-missing, constant and exactly-duplicated columns using ``X`` alone, never
``y``. A label permutation cannot change which columns are constant, so this
leaks nothing into the null and both arms are treated identically --
``AgentRCA.fit`` cleans once on its training matrix in exactly the same way. The
protocol is sound; the previous description of it was not.

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

# H3: is the agent loop's error-control advantage a property of the
# architecture, or only of the operating point it was compared at? A plain
# ranker's support pins at 1.000 because "top 40 of 474, in all 12 resamples" is
# easy. Narrow the selection depth and add resamples and the statistic has to
# spread out -- if control then matches the loop's, the advantage was the grid.
# Named by what is varied so no row in the output table is ambiguous.
VARIANTS = (
    ("univariate", 12, 5),
    ("univariate", 40, 5),
    ("univariate", 40, 10),
    ("univariate", 100, 5),
)


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


def _one(X, y, permuted, rep, kind, n_boot, select_k, top_k, arm=None):
    """One replicate: bootstrap selection frequencies, then the max statistic."""
    arm = arm or kind
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
        "arm": arm, "n_boot": int(n_boot), "select_k": int(select_k),
        "max_stability": float(supp.max()),
        "n_at_or_above_half": int((supp >= 0.5).sum()),
        "stability_values": sorted(supp[supp > 0].tolist(), reverse=True)[:60],
        # Per-sensor support in ORIGINAL index space, so a tau-thresholded
        # report can be reconstructed from the record rather than only its
        # length. See the matching field in scripts/null_fdr.py.
        "support_by_sensor": sorted(
            ([int(keep[j]), float(supp[j])] for j in np.flatnonzero(supp > 0)),
            key=lambda t: -t[1])[:60],
        "top5": top,
    }


def _heldout(null_max, real_sets, alpha, splits, rng):
    """Split-half calibrated rates, matching `scripts/abstain.py` exactly.

    tau is fitted on one half of the null replicates and every rate read off
    the other, both directions, averaged over random partitions -- because
    fitting tau and scoring abstention on the same replicates returns
    ``1 - alpha`` by construction and measures nothing.
    """
    nm = np.asarray(null_max, dtype=float)
    idx = np.arange(len(nm))
    taus, n_ab, r_ab, r_n = [], [], [], []
    for _ in range(splits):
        perm = rng.permutation(idx)
        halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
        for cal, ev in (halves, halves[::-1]):
            t = float(np.quantile(nm[cal], 1.0 - alpha))
            taus.append(t)
            n_ab.append(float(np.mean(nm[ev] < t)))
            kept = [int(sum(v >= t for v in vals)) for vals in real_sets]
            r_ab.append(float(np.mean([k == 0 for k in kept])) if kept else float("nan"))
            r_n.append(float(np.mean(kept)) if kept else float("nan"))
    return {
        "alpha": alpha,
        "tau_mean": float(np.mean(taus)),
        # the decisive column: can this statistic even be thresholded to the
        # target level, or does it saturate?
        "null_abstention_heldout": float(np.mean(n_ab)),
        "null_abstention_target": 1.0 - alpha,
        "real_abstention": float(np.mean(r_ab)),
        "real_reported_mean": float(np.mean(r_n)),
        "saturated_null_fraction": float(np.mean(nm >= 1.0)),
        "max_attainable_null_abstention": float(np.mean(nm < 1.0)),
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
    ap.add_argument("--splits", type=int, default=400,
                    help="random half/half partitions for held-out calibration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", action="store_true",
                    help="also run the H3 arms: a plain ranker at operating "
                         "points where its support cannot saturate")
    ap.add_argument("--out", default="runs/null_fdr_rankers.json")
    a = ap.parse_args()

    X, y, names = load_secom(a.root)
    n_boot = AGENT_CFG["n_boot"]
    select_k = AGENT_CFG["select_k"]
    top_k = AGENT_CFG["top_k"]
    specs = [(k, k, n_boot, select_k) for k in RANKERS]
    if a.variants:
        specs += [(f"{k} (n_boot={nb}, select_k={sk})", k, nb, sk)
                  for k, nb, sk in VARIANTS]
    jobs = [(arm, k, nb, sk, p, r) for arm, k, nb, sk in specs
            for p, n in ((True, a.null), (False, a.real))
            for r in range(n)]
    print(f"[rankers] {len(jobs)} replicates over {len(specs)} arms "
          f"(base at n_boot={n_boot} select_k={select_k} from AGENT_CFG), "
          f"{a.jobs} workers", flush=True)

    t0 = time.time()
    recs = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(_one)(X, y, p, r, k, nb, sk, top_k, arm)
        for arm, k, nb, sk, p, r in jobs)
    wall = time.time() - t0
    print(f"[rankers] done in {wall/60:.1f} min", flush=True)

    agent = None
    ap_ = Path(a.agent)
    if ap_.exists():
        agent = json.loads(ap_.read_text())

    per = {}
    for arm, kind, nb, sk in specs:
        nl = [r for r in recs if r["arm"] == arm and r["permuted"]]
        rl = [r for r in recs if r["arm"] == arm and not r["permuted"]]
        nm = [r["max_stability"] for r in nl]
        rm = [r["max_stability"] for r in rl]
        u, pv = mannwhitneyu(rm, nm, alternative="greater") if (nm and rm) \
            else (float("nan"), float("nan"))
        sep = float(np.mean([[1.0 if x > z else 0.5 if x == z else 0.0
                              for z in nm] for x in rm])) if (nm and rm) else float("nan")
        rng = np.random.default_rng(a.seed)
        ho = _heldout(nm, [r["stability_values"] for r in rl], 0.05,
                      a.splits, rng) if (nm and rl) else {}
        per[arm] = {
            "ranker": kind, "n_boot": nb, "select_k": sk,
            "is_variant": arm != kind,
            "null": _summ(nl), "real": _summ(rl),
            "prob_real_max_exceeds_null_max": sep,
            "mannwhitney_p": float(pv),
            "heldout_alpha_0.05": ho,
        }

    if agent:
        nl = [r for r in agent["records"] if r["permuted"]]
        rl = [r for r in agent["records"] if not r["permuted"]]
        nm = [r["max_stability"] for r in nl]
        rng = np.random.default_rng(a.seed)
        per["agent (full loop)"] = {
            "ranker": "agent", "n_boot": n_boot, "select_k": select_k,
            "is_variant": False,
            "null": _summ(nl), "real": _summ(rl),
            "prob_real_max_exceeds_null_max":
                agent["separation"]["prob_real_max_exceeds_null_max"],
            "mannwhitney_p": agent["separation"]["p_real_greater"],
            "heldout_alpha_0.05": _heldout(
                nm, [r["stability_values"] for r in rl], 0.05, a.splits, rng),
            "source": a.agent,
        }

    best = max(per, key=lambda k: per[k]["prob_real_max_exceeds_null_max"])
    best_control = max(
        per, key=lambda k: per[k]["heldout_alpha_0.05"].get(
            "null_abstention_heldout", float("-inf")))
    out = {
        "protocol": {
            "question": "does the agent loop's bootstrap support separate a "
                        "world with causes from one without any better than a "
                        "plain ranker's does?",
            "why_not_fdr": "any procedure that always emits a top-k has FDR 1.0 "
                           "under this null, so raw FDR cannot distinguish the "
                           "arms. Two properties can: how well the statistic "
                           "separates a world with causes from one without, "
                           "and how much error control it can actually be "
                           "thresholded to. They disagree here, so both are "
                           "reported",
            "null": "labels permuted over all wafers, class balance preserved; "
                    "X untouched",
            "matched": f"every arm uses the agent loop's own n_boot={n_boot} "
                       f"and select_k={select_k} from AGENT_CFG, and the "
                       f"identical max-over-sensors statistic, so a difference "
                       f"in separation is a difference in the ranker",
            "leakage_control": "SensorCleaner is fitted once per replicate on "
                               "the full matrix, outside the bootstrap loop. It "
                               "is unsupervised (drops all-missing, constant "
                               "and duplicate columns from X alone, never y), "
                               "so a label permutation cannot change its "
                               "output and it leaks nothing into the null. "
                               "AgentRCA.fit cleans the same way, so the arms "
                               "are matched on this too.",
            "separation_confound": "'real' replicates reuse the same labels and "
                                   "matrix, varying only bootstrap randomness, "
                                   "so P(real > null) rewards a ranker that is "
                                   "repeatable as well as one that detects "
                                   "signal. A near-deterministic ranker is "
                                   "favoured on this column relative to the "
                                   "agent loop, which carries far more internal "
                                   "stochasticity. The error-control columns do "
                                   "not share this confound.",
            "report_length_caveat": "'suspects reported' counts survivors of "
                                    "tau from different candidate universes -- "
                                    "all surviving sensors for a plain ranker, "
                                    "correlation-group representatives admitted "
                                    "to verification for the agent loop. It is "
                                    "a power/cost trade-off, not an accuracy "
                                    "axis: without ground-truth causes a longer "
                                    "list is not self-evidently better.",
            "splits_caveat": "repeated partitions reduce partition noise; they "
                             "do not create new evidence beyond the "
                             "n_null + n_real replicates actually run",
            "n_null": a.null, "n_real": a.real, "top_k": top_k,
            "agent_arm_source": a.agent if agent else None,
        },
        "environment": {"python": platform.python_version(),
                        "sklearn": sklearn.__version__,
                        "numpy": np.__version__,
                        "platform": platform.platform(),
                        "wall_min": wall / 60.0},
        "best_separating_arm": best,
        "best_controlled_arm": best_control,
        "note_on_saturation": "a support statistic that saturates at 1.0 on a "
                              "share of *null* replicates cannot be thresholded "
                              "to an arbitrary level: no tau <= 1 excludes "
                              "those replicates, so max_attainable_null_"
                              "abstention caps the error control that arm can "
                              "offer under the rule actually implemented here "
                              "(report iff support >= tau, tau <= 1). Two "
                              "escapes exist and neither is used: tau > 1 "
                              "abstains always, at zero power, and a randomised "
                              "boundary rule could interpolate intermediate "
                              "levels. The cap is a property of this discrete "
                              "max statistic at this bootstrap count and "
                              "selection depth, not of the ranker itself.",
        "per_ranker": per,
        "records": recs,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'arm':<40}{'P(real>null)':>13}{'tau':>7}{'null ctrl':>11}"
          f"{'ceiling':>9}{'suspects':>10}{'abstains':>10}")
    for kind, v in sorted(per.items(),
                          key=lambda kv: -kv[1]["prob_real_max_exceeds_null_max"]):
        h = v["heldout_alpha_0.05"]
        print(f"{kind:<40}{v['prob_real_max_exceeds_null_max']:>13.3f}"
              f"{h['tau_mean']:>7.3f}{h['null_abstention_heldout']:>11.1%}"
              f"{h['max_attainable_null_abstention']:>9.1%}"
              f"{h['real_reported_mean']:>10.2f}"
              f"{h['real_abstention']:>10.0%}")
    print("\n  'null ctrl' = held-out share of no-cause worlds correctly kept "
          "silent (target 95%)")
    print("  'ceiling'   = the most any threshold could ever achieve, given "
          "how often the statistic saturates")
    print(f"\nbest separating arm: {best}")
    print(f"best error-controlled arm: {best_control}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
