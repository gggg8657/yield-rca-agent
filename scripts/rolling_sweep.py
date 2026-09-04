"""Does the forward-in-time ordering hold at more than one block count?

    OMP_NUM_THREADS=1 python scripts/rolling_sweep.py --jobs 16

``eval_secom.py`` runs the rolling-origin protocol at one block count, and the
resulting "sparse arms do better forward in time" is the weakest claim in the
report -- four origins, wide intervals. Rather than write around that, this
sweeps the block count: fewer blocks means larger, less noisy test sets but
fewer origins; more blocks means the opposite. If the ordering is real it
should survive the trade, and if it is an artefact of one partition it should
not.

The origins within a block count share training data, so the intervals here are
optimistic in the usual way and the script says so in its output. What it can
settle is whether the *sign* is stable. Writes ``runs/rolling_sweep.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from arms import SEED, agent, hgb_all, rf_all, univar_rf
from yieldrca.data import load_secom
from yieldrca.evaluate import mean_ci, paired_delta, rolling_origin_splits

ARMS = {
    "rf_all": rf_all,
    "hgb_all": hgb_all,
    "univar_top25_rf": univar_rf,
    "agent_rf": (lambda: agent("rf")),
}


def _one(name, factory, X, y, tr, te, blocks, origin):
    est = clone(factory())
    est.fit(X[tr], y[tr])
    s = est.predict_proba(X[te])
    s = s[:, 1] if np.ndim(s) == 2 else s
    return {"arm": name, "blocks": blocks, "origin": origin,
            "auc": float(roc_auc_score(y[te], s)),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_fail_test": int(y[te].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, nargs="*", default=[3, 4, 5, 8, 10])
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/rolling_sweep.json")
    a = ap.parse_args()

    X, y, names, t = load_secom(a.root, with_time=True)
    jobs = []
    for b in a.blocks:
        for k, (tr, te) in enumerate(rolling_origin_splits(t, b)):
            if y[te].sum() < 3:      # an AUC on <3 positives is not a number
                continue
            for name, f in ARMS.items():
                jobs.append((name, f, tr, te, b, k))

    t0 = time.time()
    recs = list(Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(_one)(n, f, X, y, tr, te, b, k) for n, f, tr, te, b, k in jobs))
    wall = (time.time() - t0) / 60.0

    by = {}
    for r in recs:
        by.setdefault(r["blocks"], {}).setdefault(r["arm"], []).append(r)
    out = {
        "protocol": "rolling origin (expanding window) at several block "
                    "counts; origins within a block count share training "
                    "data, so the intervals are optimistic -- what this "
                    "settles is whether the sign is stable",
        "min_fails_in_test_block": 3,
        "wall_min": wall,
        "block_counts": sorted(by),
        "per_block_count": {},
        "records": recs,
    }
    for b in sorted(by):
        d = by[b]
        arms = sorted(d, key=lambda k_: -float(np.mean([r["auc"] for r in d[k_]])))
        entry = {
            "n_origins": len(next(iter(d.values()))),
            "test_block_fails": [r["n_fail_test"]
                                 for r in next(iter(d.values()))],
            "per_arm": {k_: mean_ci([r["auc"] for r in d[k_]]) for k_ in d},
            "best_arm": arms[0],
        }
        if "agent_rf" in d and "rf_all" in d:
            order = lambda rs: [r["auc"] for r in sorted(rs, key=lambda r: r["origin"])]
            entry["agent_vs_rf_all"] = paired_delta(order(d["agent_rf"]),
                                                    order(d["rf_all"]))
        out["per_block_count"][str(b)] = entry

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'blocks':>6} {'origins':>7} {'best arm':<18} "
          f"{'agent_rf - rf_all':>26}  per-arm means")
    for b in sorted(by):
        e = out["per_block_count"][str(b)]
        d = e.get("agent_vs_rf_all")
        dd = (f"{d['mean']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
              if d else "-")
        means = "  ".join(f"{k}={v['mean']:.3f}"
                          for k, v in sorted(e["per_arm"].items(),
                                             key=lambda kv: -kv[1]["mean"]))
        print(f"{b:>6} {e['n_origins']:>7} {e['best_arm']:<18} {dd:>26}  {means}")
    print(f"\nwrote {a.out} ({wall:.1f} min)")


if __name__ == "__main__":
    main()
