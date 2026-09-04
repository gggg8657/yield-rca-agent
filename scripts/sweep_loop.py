"""Sensitivity of the agent loop to its own structural settings.

    OMP_NUM_THREADS=1 python scripts/sweep_loop.py --repeats 2 --jobs 16

The operating point in ``arms.py`` is pre-registered rather than tuned, which
is only defensible if the whole surface around it is published. This sweeps the
attribution mode, the candidate-pool size and the (vote depth, threshold,
cap) triple that controls how many sensors survive, against the plain
random-forest baseline on identical folds, and writes
``runs/secom_loop_sweep.json``.

The limit row -- threshold 0, no cap -- is the sanity check: with the drop step
disabled the loop should converge back onto the baseline, so any gap there is a
bug, not a finding.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from arms import SEED, agent, rf_all, univar_rf
from yieldrca.data import load_secom
from yieldrca.evaluate import Arm, repeated_cv, summarize

GRID = [
    # tag, overrides
    ("sparse",     dict(n_screen=150, select_k=20, stability_min=0.5, max_select=25)),
    ("operating",  dict(n_screen=150, select_k=40, stability_min=0.3, max_select=25)),
    ("wide",       dict(n_screen=150, select_k=60, stability_min=0.2, max_select=60)),
    ("loose",      dict(n_screen=250, select_k=100, stability_min=0.15, max_select=150)),
    ("no_drop",    dict(n_screen=474, select_k=474, stability_min=0.0, max_select=474)),
]


def _extras(est):
    return {"n_selected": int(len(est.selected_)),
            "n_candidates": int(est.n_candidates_)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/secom_loop_sweep.json")
    a = ap.parse_args()

    X, y, names = load_secom(a.root)
    arms = [
        Arm("rf_all", rf_all, "random forest, all sensors", "baseline"),
        Arm("univar_top25_rf", univar_rf, "univariate top-25 -> RF", "control"),
    ]
    for tag, over in GRID:
        for attr in ("permutation", "model"):
            arms.append(Arm(
                f"agent_{tag}_{attr}",
                (lambda o=over, at=attr: agent("rf", attribution=at, **o)),
                f"agent loop [{tag}] attribution={attr}", "agent", extras=_extras,
                meta={"tag": tag, "attribution": attr, **over}))

    t0 = time.time()
    recs, _ = repeated_cv(arms, X, y, n_splits=a.splits, n_repeats=a.repeats,
                          seed=SEED, n_jobs=a.jobs)
    s = summarize(recs, "auc")
    nsel = {}
    for r in recs:
        if "n_selected" in r:
            nsel.setdefault(r["arm"], []).append(r["n_selected"])

    out = {
        "protocol": f"RepeatedStratifiedKFold({a.splits} x {a.repeats}, "
                    f"seed {SEED}); identical folds for every row",
        "wall_min": (time.time() - t0) / 60.0,
        "grid": {tag: over for tag, over in GRID},
        "arms": {x.name: {"label": x.label, "kind": x.kind, "meta": x.meta}
                 for x in arms},
        "auc": s,
        "n_selected_mean": {k: float(np.mean(v)) for k, v in nsel.items()},
        "records": recs,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'arm':<32} {'AUC':>6} {'95% CI':>16} {'n_sel':>6}  vs rf_all")
    for name, v in sorted(s["per_arm"].items(), key=lambda kv: -kv[1]["mean"]):
        d = s["paired"].get(f"{name}__vs__rf_all")
        ci = f"[{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
        dd = (f"{d['mean']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
              if d else "-")
        print(f"{name:<32} {v['mean']:.3f} {ci:>16} "
              f"{out['n_selected_mean'].get(name, float('nan')):6.1f}  {dd}")
    print(f"\nwrote {a.out} ({out['wall_min']:.1f} min)")


if __name__ == "__main__":
    main()
