"""H10: is the loop's shorter suspect list deduplication or under-reporting?

Every turn so far has scored report length as though longer were better. But the
loop runs a `CorrelatorAgent` whose job is to collapse near-identical sensors to
one representative, and 179 SECOM sensors have a partner correlated above 0.99.
A shorter list from a deduplicating ranker may be the same information with the
duplicates removed.

Measured at **matched list length** -- every arm's stored real-label `top5` --
so the comparison is about the ranking rather than about how long each arm's
report happens to be. Family membership uses `yieldrca.stability.cluster_map`
at the same 0.99 threshold as the cluster-aware stability column.

No model is fitted: this reads `top5` sets already recorded by earlier runs.

    python scripts/dedup.py
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from yieldrca.data import load_secom
from yieldrca.stability import cluster_map


def _sets(root: Path):
    """(label, [top5 sets]) for every arm with real-label replicates on disk."""
    out = []
    for fn, lab in (("null_fdr_k5_model_b40.json",
                     "agent loop (select_k=5, model, n_boot=40)"),
                    ("null_fdr_k5_model.json",
                     "agent loop (select_k=5, model, n_boot=12)"),
                    ("null_fdr_k5.json",
                     "agent loop (select_k=5, permutation, n_boot=12)")):
        p = root / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        t = [r["top5"] for r in d["records"]
             if not r["permuted"] and len(r.get("top5", [])) == 5]
        if t:
            out.append((lab, t))
    rk = root / "null_fdr_rankers.json"
    if rk.exists():
        d = json.loads(rk.read_text())
        for name, v in sorted((d["per_ranker"] or {}).items(),
                              key=lambda kv: kv[1].get("n_boot", 0)):
            t = [r["top5"] for r in d["records"]
                 if r["arm"] == name and not r["permuted"]
                 and len(r.get("top5", [])) == 5]
            if t:
                out.append((name, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--root", default="data")
    ap.add_argument("--thresh", default="0.90,0.95,0.99",
                    help="sweep. 0.90 is the loop's own "
                         "corr_thresh; 0.99 is the map the "
                         "stability column uses and the one "
                         "H10 was (wrongly) pre-registered at")
    ap.add_argument("--out", default="runs/dedup.json")
    a = ap.parse_args()

    X, y, _ = load_secom(a.root)
    threshes = [float(v) for v in str(a.thresh).split(",")]
    by_thresh, verdicts = {}, {}
    sets_all = _sets(Path(a.runs))

    for th in threshes:
        cmap, _n_groups = cluster_map(X, thresh=th)
        n_fam = len(set(cmap.values()))
        print(f"\n[dedup] |r| >= {th}: {n_fam} correlation families", flush=True)
        arms = {}
        for label, sets in sets_all:
            fams = np.array([len({cmap[int(j)] for j in s}) for s in sets],
                            dtype=float)
            arms[label] = {
                "n_replicates": len(sets),
                "families_mean": float(fams.mean()),
                "families_sd": (float(fams.std(ddof=1)) if len(fams) > 1
                                else 0.0),
                "families_se": (float(fams.std(ddof=1) / np.sqrt(len(fams)))
                                if len(fams) > 1 else 0.0),
                "families_min": int(fams.min()),
                "families_max": int(fams.max()),
                "constant_across_replicates": bool(fams.min() == fams.max()),
            }
            print(f"  {label:52s} {fams.mean():.3f} "
                  f"(min {int(fams.min())}, max {int(fams.max())})", flush=True)

        loop_k = "agent loop (select_k=5, model, n_boot=40)"
        uni_k = next((k for k in arms
                      if k.startswith("univariate (n_boot=40, select_k=5")),
                     None)
        if loop_k in arms and uni_k:
            lo, un = arms[loop_k], arms[uni_k]
            diff = lo["families_mean"] - un["families_mean"]
            se = float(np.hypot(lo["families_se"], un["families_se"]))
            both_const = (lo["constant_across_replicates"]
                          and un["constant_across_replicates"])
            verdicts[str(th)] = {
                "loop": loop_k, "univariate": uni_k,
                "loop_families": lo["families_mean"],
                "univariate_families": un["families_mean"],
                "difference": diff, "se_of_difference": se,
                "both_constant_across_replicates": both_const,
                "holds": bool(diff > se or (both_const and diff > 0)),
                "note": ("both arms are constant across every replicate, so "
                         "the standard-error bar is degenerate and the honest "
                         "statement is the replicate count, not a p-value"
                         if both_const else ""),
            }
            v = verdicts[str(th)]
            print(f"  -> loop {lo['families_mean']:.3f} vs univariate "
                  f"{un['families_mean']:.3f}, diff {diff:+.3f}; "
                  f"{'HOLDS' if v['holds'] else 'REFUTED'}"
                  f" ({lo['n_replicates']} replicates)", flush=True)
        by_thresh[str(th)] = {"n_families": int(n_fam), "arms": arms}

    out = {
        "protocol": {
            "hypothesis": "H10. The loop's shorter suspect list is "
                          "deduplication, not under-reporting.",
            "measured_on": "each arm's stored real-label top5 -- matched list "
                           "length, so this is a statement about the ranking "
                           "and not about how long each arm's report is",
            "families": "connected components at |r| >= thresh from "
                        "yieldrca.stability.cluster_map",
            "threshold_note": "H10 was pre-registered at 0.99, the map the "
                              "cluster-aware stability column uses. That was "
                              "the wrong criterion: the loop's CorrelatorAgent "
                              "groups at corr_thresh = 0.90, so 0.99 tests a "
                              "rule it never claimed to enforce. Both are "
                              "reported and the pre-registered one is named as "
                              "pre-registered.",
            "bar": "pre-registered in critique_log.md: the difference must "
                   "exceed the standard error of the difference, and a tie is "
                   "not a win for the architecture",
            "cannot_settle": "the tau-thresholded reported sets are not "
                             "reconstructible from disk (stability_values are "
                             "stored without sensor identities), so this does "
                             "not measure the reports themselves",
            "thresholds": threshes,
            "no_model_fitted": True,
        },
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
        "by_threshold": by_thresh,
        "verdicts": verdicts,
        "preregistered_threshold": 0.99,
        "loop_corr_thresh": 0.90,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[dedup] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
