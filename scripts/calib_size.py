"""How much of the gap to nominal is the grid, and how much is calibration?

`runs/null_fdr_rankers.json` shows error control lives on a grid of spacing
1/`n_boot`, and that nominal 0.95 becomes attainable once `n_boot` >= 40. But
at `n_boot` = 100 the grid contains 0.950 exactly and the measured control is
94.1% -- so something other than the grid is also costing accuracy.

That something is the calibration. tau is a quantile estimated from a finite
number of null replicates, and a finer grid needs more of them to resolve. This
script separates the two costs, using **no model fits at all**: every quantity
is a resampling of the per-replicate `max_stability` values those runs already
recorded.

For a given calibration size *m*: draw *m* null replicates, fit tau at alpha,
evaluate the resulting rule on the held-out remainder, repeat, average. Against
that, the **oracle** rule -- the best attainable grid point, i.e. what an
infinite calibration set would deliver -- is computed from all replicates.

    gap to nominal  =  (nominal - oracle)      <- grid resolution
                    +  (oracle  - measured)    <- calibration noise

`--m 100` reproduces `scripts/abstain.py`'s half-of-200 protocol, which is
asserted as a self-check rather than assumed.

    python scripts/calib_size.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ALPHA = 0.05


def _arms(root: Path):
    """(label, n_boot, null max-statistics) for every arm with records."""
    out = []
    rk = root / "null_fdr_rankers.json"
    if rk.exists():
        d = json.loads(rk.read_text())
        per, recs = d["per_ranker"], d["records"]
        for name, v in sorted(per.items(), key=lambda kv: kv[1].get("n_boot", 0)):
            if not (v.get("is_variant") and v.get("select_k") == 5
                    and v.get("ranker") == "univariate"):
                continue
            mx = [r["max_stability"] for r in recs
                  if r["arm"] == name and r["permuted"]]
            if mx:
                out.append((f"univariate, select_k=5, n_boot={v['n_boot']}",
                            v["n_boot"], np.asarray(mx, dtype=float)))
    for fn, lab in (("null_fdr_k5_model.json",
                     "agent loop, select_k=5, model, n_boot=12"),
                    ("null_fdr_k5_model_b40.json",
                     "agent loop, select_k=5, model, n_boot=40"),
                    ("null_fdr_k5.json",
                     "agent loop, select_k=5, permutation, n_boot=12")):
        p = root / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        nb = d["protocol"]["agent_cfg"]["n_boot"]
        mx = [r["max_stability"] for r in d["records"] if r["permuted"]]
        if mx:
            out.append((lab, nb, np.asarray(mx, dtype=float)))
    return out


def attainable(mx, n_boot):
    """The control values a threshold on this grid can produce."""
    return sorted({float(1.0 - (mx >= k / n_boot).mean())
                   for k in range(1, n_boot + 1)}, reverse=True)


def oracle(mx, n_boot, target):
    """Best attainable control, i.e. what an infinite calibration would give.

    "Best" is the attainable value closest to the target -- not the largest,
    since overshooting nominal costs power on real labels just as undershooting
    costs error control.
    """
    ach = attainable(mx, n_boot)
    return min(ach, key=lambda a: abs(a - target))


def measured(mx, m, alpha, reps, rng):
    """Fit tau on m replicates, evaluate on the rest, averaged over `reps`."""
    n = len(mx)
    if m >= n:
        return float("nan"), float("nan"), 0
    ctl = []
    for _ in range(reps):
        idx = rng.permutation(n)
        cal, ev = mx[idx[:m]], mx[idx[m:]]
        tau = float(np.quantile(cal, 1.0 - alpha))
        ctl.append(float((ev < tau).mean()))
    return float(np.mean(ctl)), float(np.std(ctl, ddof=1)), n - m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--sizes", default="25,50,100,150")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/calib_size.json")
    a = ap.parse_args()

    root = Path(a.runs)
    sizes = [int(v) for v in a.sizes.split(",")]
    target = 1.0 - a.alpha

    arms = {}
    for label, nb, mx in _arms(root):
        # A per-arm RNG derived from the label, not one shared generator.
        # With a shared generator the draws for every arm depend on how many
        # arms preceded it, so adding an arm silently moved another arm's
        # published number by 0.004. Seeding from the label makes each row
        # reproducible on its own.
        rng = np.random.default_rng(
            [a.seed, int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],
                                    "big")])
        orc = oracle(mx, nb, target)
        curve = {}
        for m in sizes:
            mean, sd, n_ev = measured(mx, m, a.alpha, a.reps, rng)
            curve[str(m)] = {"m": m, "control_mean": mean, "control_sd": sd,
                             "n_heldout": n_ev,
                             "calibration_loss": (orc - mean
                                                  if np.isfinite(mean)
                                                  else float("nan"))}
        arms[label] = {
            "n_boot": nb,
            "n_null_replicates": int(len(mx)),
            "p_saturated": float((mx >= 1.0).mean()),
            "attainable_above_060": [a_ for a_ in attainable(mx, nb)
                                     if a_ > 0.60],
            "oracle_control": orc,
            "grid_gap": target - orc,
            "curve": curve,
        }
        print(f"{label}", flush=True)
        print(f"   oracle {orc:.3f}  (grid gap {target - orc:+.3f})", flush=True)
        for m in sizes:
            c = curve[str(m)]
            if np.isfinite(c["control_mean"]):
                print(f"   m={m:4d}  control {c['control_mean']:.3f} "
                      f"+-{c['control_sd']:.3f}  calibration loss "
                      f"{c['calibration_loss']:+.3f}  (held out "
                      f"{c['n_heldout']})", flush=True)

    out = {
        "protocol": {
            "question": "Split the gap between nominal control and delivered "
                        "control into a grid term and a calibration term.",
            "source": "per-replicate max_stability already recorded by "
                      "null_fdr_rankers.py and null_fdr.py; no model is "
                      "fitted here",
            "oracle": "the attainable grid value closest to nominal -- what an "
                      "infinite calibration set would deliver. Closest rather "
                      "than largest, because overshooting nominal costs power "
                      "on real labels.",
            "measured": f"fit tau on m null replicates, evaluate on the "
                        f"held-out remainder, averaged over {a.reps} draws",
            "alpha": a.alpha, "nominal": target, "reps": a.reps,
            "seed": a.seed,
            "cross_check": "m=100 on a 200-replicate arm is abstain.py's "
                           "half-and-half protocol, so those rows should "
                           "reproduce its published control",
        },
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__},
        "sizes": sizes,
        "arms": arms,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
