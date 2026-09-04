"""Is the chronological collapse actually drift? Measure it, don't assert it.

    OMP_NUM_THREADS=1 python scripts/drift.py --jobs 16

The rolling-origin table shows every arm losing most of its skill when it has
to predict the next block of wafers, and "the sensors drift" is the obvious
explanation. It is also the kind of explanation that gets written into a README
without being checked, so this script checks it three ways and writes
``runs/drift.json``:

1. **Adversarial validation.** Label each wafer by *when* it was made -- early
   70% vs late 30% -- and cross-validate a classifier on the sensors alone. If
   a model can recover the era from the process data, the training and test
   distributions of the chronological split are not the same distribution, and
   an AUC near 1.0 says so about as loudly as it can be said.
2. **Per-sensor two-sample distance.** A Kolmogorov-Smirnov statistic per
   sensor between the first and last time block, with Benjamini-Hochberg
   control over 474 tests, so "how many sensors moved" has a number.
3. **Label drift.** The fail rate per time block. If the *prior* moves, part of
   the collapse is not covariate shift at all, and the two need separating.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from arms import SEED, rf_all
from yieldrca.data import load_secom
from yieldrca.evaluate import mean_ci
from yieldrca.preprocess import SensorCleaner


def bh_reject(pvals, alpha=0.01):
    """Benjamini-Hochberg: how many of these p-values survive at FDR alpha."""
    p = np.sort(np.asarray(pvals))
    m = len(p)
    if m == 0:
        return 0, 1.0
    thresh = alpha * np.arange(1, m + 1) / m
    below = np.flatnonzero(p <= thresh)
    if len(below) == 0:
        return 0, 0.0
    k = below[-1] + 1
    return int(k), float(p[k - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--late-frac", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="runs/drift.json")
    a = ap.parse_args()

    X, y, names, t = load_secom(a.root, with_time=True)
    order = np.argsort(t, kind="stable")
    X, y, t = X[order], y[order], t[order]
    n = len(y)

    # -- 1. adversarial validation: can the sensors tell you the era? --------
    cut = int(round((1 - a.late_frac) * n))
    era = np.zeros(n, dtype=int)
    era[cut:] = 1
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    adv = cross_val_score(rf_all(), X, era, cv=cv, scoring="roc_auc",
                          n_jobs=a.jobs)
    adv_summary = mean_ci(adv)

    # a control: the same test against a *shuffled* era label, which must
    # land at chance -- otherwise the number above means nothing
    rng = np.random.default_rng(SEED)
    adv_ctl = cross_val_score(rf_all(), X, era[rng.permutation(n)], cv=cv,
                              scoring="roc_auc", n_jobs=a.jobs)

    # -- 2. per-sensor KS between the first and last time block --------------
    cl = SensorCleaner().fit(X)
    Xc = cl.transform(X)
    blocks = np.array_split(np.arange(n), a.blocks)
    first, last = blocks[0], blocks[-1]
    ks, pv = [], []
    for j in range(Xc.shape[1]):
        a_ = Xc[first, j][~np.isnan(Xc[first, j])]
        b_ = Xc[last, j][~np.isnan(Xc[last, j])]
        if len(a_) < 10 or len(b_) < 10:
            ks.append(np.nan)
            pv.append(1.0)
            continue
        r = stats.ks_2samp(a_, b_)
        ks.append(float(r.statistic))
        pv.append(float(r.pvalue))
    ks = np.asarray(ks)
    pv = np.asarray(pv)
    n_rej, p_cut = bh_reject(pv, a.alpha)
    finite = ks[np.isfinite(ks)]

    # -- 3. label drift: fail rate per block --------------------------------
    per_block = [{"block": i, "n": int(len(b)),
                  "n_fail": int(y[b].sum()), "fail_rate": float(y[b].mean())}
                 for i, b in enumerate(blocks)]
    rates = [b["fail_rate"] for b in per_block]
    counts = np.array([[b["n_fail"], b["n"] - b["n_fail"]] for b in per_block])
    chi2 = stats.chi2_contingency(counts)

    out = {
        "protocol": {
            "adversarial": f"label = wafer in the last {a.late_frac:.0%} by "
                           f"time; RepeatedStratifiedKFold(5 x 3, seed {SEED}) "
                           f"on the sensors alone, same pipeline as `rf_all`",
            "ks": f"KS_2samp between time block 0 and block {a.blocks - 1} "
                  f"per surviving sensor, Benjamini-Hochberg at "
                  f"FDR {a.alpha}",
            "label_drift": f"fail rate over {a.blocks} contiguous time blocks, "
                           f"chi-square test of homogeneity",
        },
        "adversarial": {
            "auc": adv_summary,
            "auc_shuffled_control": mean_ci(adv_ctl),
            "n_early": int(cut), "n_late": int(n - cut),
        },
        "sensor_drift": {
            "n_sensors_tested": int(np.isfinite(ks).sum()),
            "n_significant": n_rej,
            "frac_significant": float(n_rej / max(np.isfinite(ks).sum(), 1)),
            "bh_p_cutoff": p_cut,
            "ks_median": float(np.median(finite)),
            "ks_p90": float(np.percentile(finite, 90)),
            "ks_max": float(finite.max()),
        },
        "label_drift": {
            "per_block": per_block,
            "fail_rate_min": float(min(rates)),
            "fail_rate_max": float(max(rates)),
            "chi2": float(chi2.statistic),
            "chi2_p": float(chi2.pvalue),
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "protocol"},
                     indent=2)[:2000])
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
