"""What would it cost the pipeline to be allowed to say nothing?

`runs/null_fdr.json` establishes that the agent loop reports root causes on
permuted labels, so its false-discovery rate under a no-causal-sensor null is
1.0. This script turns that null into a decision rule and prices it.

**The rule.** Let ``s_j`` be suspect *j*'s bootstrap selection frequency and let

    tau(alpha) = the (1 - alpha) quantile of max_j s_j across null replicates.

Report only suspects with ``s_j >= tau(alpha)``, and report nothing when none
qualify. Thresholding the *maximum* rather than each sensor separately is
Westfall-Young max-statistic control: it is family-wise over the sensors a
replicate screens, it needs no independence assumption between sensors -- which
matters here, since 179 SECOM sensors have a partner correlated above 0.99 --
and it is calibrated on the pipeline's own bootstrap rather than on an
asymptotic argument that bootstrap does not satisfy.

**Why the calibration is split.** Choosing tau on a set of null replicates and
then measuring the abstention rate on those same replicates returns
``1 - alpha`` by construction, which measures nothing. The null replicates are
therefore halved: tau is fitted on one half and every rate is reported on the
other, which the held-out half has never influenced. Both halves are also
swapped and the pair averaged, and the split is repeated over many random
partitions so the reported figure is not an artifact of one arbitrary cut.

No model is refitted here. Every quantity is a function of the per-replicate
suspect supports already recorded in `runs/null_fdr.json`, so this costs
seconds and inherits that run's protocol exactly.

    python scripts/abstain.py
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

ALPHAS = (0.10, 0.05, 0.01)


def _tau(cal_max, alpha):
    return float(np.quantile(np.asarray(cal_max, dtype=float), 1.0 - alpha))


def _mean_pairwise_overlap(sets, k):
    """Mean |A n B| / k over all unordered pairs; the stability definition."""
    import itertools
    pairs = [len(a & b) / k for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(pairs)) if pairs else float("nan")


def null_structure(null, real, n_eff):
    """Do null replicates keep re-finding the *same* noise sensors?

    Answers the one alternative explanation that would deflate the whole
    false-discovery result: "permuting labels leaves the sensor correlation
    structure intact, so the loop is reporting that structure rather than
    inventing anything, and the null is unfairly easy." If that held, null
    replicates would agree with each other about which sensors to name. If
    instead they agree at roughly the rate two random top-5 lists would, the
    loop is resampling noise, and the false discoveries really are fresh each
    time.

    ``n_eff`` is the number of sensors in play, so ``k / n_eff`` is the
    chance-level overlap for a uniformly random top-k.
    """
    out = {"n_eff_sensors": int(n_eff)}
    for label, reps in (("null", null), ("real", real)):
        top5 = [set(r["top5"]) for r in reps if len(r.get("top5", [])) == 5]
        sel = [set(r["selected"]) for r in reps if r.get("selected")]
        out[label] = {
            "n_replicates_top5": len(top5),
            "top5_pairwise_overlap": _mean_pairwise_overlap(top5, 5),
            "selected_pairwise_overlap": (
                float(np.mean([len(a & b) / max(len(a | b), 1)
                               for i, a in enumerate(sel)
                               for b in sel[i + 1:]]))
                if len(sel) > 1 else float("nan")),
            "distinct_sensors_ever_named": len(set().union(*sel)) if sel else 0,
        }
    out["random_floor_top5"] = 5.0 / n_eff if n_eff else float("nan")
    o_null = out["null"]["top5_pairwise_overlap"]
    o_real = out["real"]["top5_pairwise_overlap"]
    floor = out["random_floor_top5"]
    out["verdict"] = (
        "null replicates agree with each other at {:.3f}, against a random "
        "floor of {:.3f} and {:.3f} on real labels".format(o_null, floor, o_real)
        + (" -- indistinguishable from resampled noise, so the "
           "correlation-structure explanation does not hold"
           if o_null <= 3 * floor else
           " -- well above the random floor, so part of what the loop names on "
           "the null is the sensor correlation structure rather than fresh "
           "noise, and the false-discovery figure should be read with that in "
           "mind"))
    return out


def _evaluate(reps, tau):
    """Abstention rate and surviving-suspect count for one set of replicates.

    ``fallback_reached_report`` answers an objection raised against this
    repository's write-up: the cross-tab in `RESULTS.md` shows that no
    replicate the never-empty guard fired on clears the *full-null* tau, which
    is not the tau this function is called with. So count it here instead,
    inside the split-half protocol that produces the headline figure -- the
    number of evaluation-half replicates that both had the guard fire and named
    at least one suspect over that split's own tau. If it is 0 for every split,
    the guard provably never reaches the calibrated report.
    """
    kept = [int(sum(v >= tau for v in r["stability_values"])) for r in reps]
    reach = [k > 0 and r.get("fallback_fired", False)
             for k, r in zip(kept, reps)]
    return {
        "n": len(reps),
        "abstention_rate": float(np.mean([k == 0 for k in kept])) if kept else float("nan"),
        "n_reported_mean": float(np.mean(kept)) if kept else float("nan"),
        "n_reported_max": int(max(kept)) if kept else 0,
        "fallback_reached_report": int(sum(reach)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/null_fdr.json")
    ap.add_argument("--splits", type=int, default=400,
                    help="random half/half partitions of the null replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eff", type=int, default=474,
                    help="sensors surviving cleaning, for the random floor")
    ap.add_argument("--out", default="runs/abstain.json")
    a = ap.parse_args()

    src = json.loads(Path(a.src).read_text())
    null = [r for r in src["records"] if r["permuted"]]
    real = [r for r in src["records"] if not r["permuted"]]
    if len(null) < 4:
        raise SystemExit(f"{a.src}: need at least 4 null replicates, "
                         f"found {len(null)}")
    print(f"[abstain] {len(null)} null and {len(real)} real replicates from "
          f"{a.src}", flush=True)

    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(null))
    rows = {}
    for alpha in ALPHAS:
        taus, null_ab, null_fd, real_ab, real_n = [], [], [], [], []
        reach = []
        for _ in range(a.splits):
            perm = rng.permutation(idx)
            halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
            for cal, ev in (halves, halves[::-1]):
                t = _tau([null[i]["max_stability"] for i in cal], alpha)
                e_null = _evaluate([null[i] for i in ev], t)
                e_real = _evaluate(real, t)
                taus.append(t)
                null_ab.append(e_null["abstention_rate"])
                null_fd.append(e_null["n_reported_mean"])
                real_ab.append(e_real["abstention_rate"])
                real_n.append(e_real["n_reported_mean"])
                reach.append(e_null["fallback_reached_report"])
        rows[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "tau_mean": float(np.mean(taus)),
            "tau_sd": float(np.std(taus, ddof=1)),
            "tau_min": float(np.min(taus)),
            "tau_max": float(np.max(taus)),
            # Does the never-empty guard ever reach the calibrated report?
            # Counted inside every calibration/evaluation split rather than
            # against one full-null threshold.
            "n_splits_evaluated": len(reach),
            "fallback_reached_report_total": int(sum(reach)),
            "fallback_reached_report_rate": (
                float(np.mean([x > 0 for x in reach])) if reach
                else float("nan")),
            # held-out null: the share of no-cause worlds correctly kept silent
            "null_abstention_heldout": float(np.mean(null_ab)),
            "null_abstention_target": 1.0 - alpha,
            "null_false_discoveries_heldout": float(np.mean(null_fd)),
            # real labels: the price of the guarantee
            "real_abstention": float(np.mean(real_ab)),
            "real_reported_mean": float(np.mean(real_n)),
        }
        r = rows[f"alpha_{alpha}"]
        print(f"  alpha {alpha:<5} tau {r['tau_mean']:.3f}  "
              f"null abstains {r['null_abstention_heldout']:.1%} "
              f"(target {1-alpha:.0%})  "
              f"real abstains {r['real_abstention']:.1%}  "
              f"real suspects {r['real_reported_mean']:.2f}", flush=True)

    n_eff = a.n_eff
    struct = null_structure(null, real, n_eff)
    print(f"\n[abstain] {struct['verdict']}", flush=True)

    baseline = {
        "null_abstention": src["null"]["abstention_rate"],
        "null_reported_mean": src["null"]["n_reported_mean"],
        "real_reported_mean": src["real"]["n_reported_mean"],
        "note": "the pipeline as it stands, with no abstention rule",
    }
    out = {
        "protocol": {
            "rule": "report suspect j iff its bootstrap support s_j >= "
                    "tau(alpha), where tau(alpha) is the (1-alpha) quantile of "
                    "max_j s_j over null replicates (Westfall-Young "
                    "max-statistic, family-wise over the sensors screened)",
            "calibration": f"null replicates split in half; tau fitted on one "
                           f"half, all rates reported on the other; both "
                           f"directions averaged over {a.splits} random "
                           f"partitions",
            "why_split": "fitting tau and measuring abstention on the same "
                         "replicates returns 1-alpha by construction and "
                         "measures nothing",
            "source": a.src,
            "n_null": len(null), "n_real": len(real),
            "refits": 0,
            # carried through so a report can say which operating point these
            # rates belong to without re-opening the source JSON
            "select_k": ((src.get("protocol") or {}).get("agent_cfg") or {})
                        .get("select_k"),
            "n_boot": ((src.get("protocol") or {}).get("agent_cfg") or {})
                      .get("n_boot"),
        },
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__,
                        "platform": platform.platform()},
        "no_rule_baseline": baseline,
        "null_structure": struct,
        "levels": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
