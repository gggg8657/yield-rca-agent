"""Is a reported suspect's association with failure invariant across production periods?

Permutation importance is not causal. It ranks sensors by how much a *model*
leans on them, which is a statement about the model and the sample, not about
the process. This script asks the weakest question that has an identification
argument attached, and answers it honestly.

**The argument.** Invariant Causal Prediction (Peters, Buhlmann & Meinshausen,
JRSS-B 2016) takes environments in which the causal mechanism is unchanged but
the covariate distribution moves, and keeps only predictors whose relationship
with the response is stable across them. A sensor whose association with failure
appears in one production period and vanishes or reverses in the next cannot be
a stable cause of failure; something else moved. SECOM's contiguous time blocks
are candidate environments, and `runs/drift.json` already establishes that they
*are* different environments -- adversarial validation separates them at AUC
0.9926.

**What this is not.** Full ICP tests subsets and returns a confidence set for
the causal parents; with 474 sensors and 104 fails that is neither computable
nor powered. This is the per-sensor marginal screen: a *necessary* condition for
a sensor to be a stable cause, not a sufficient one. Passing does not make a
sensor causal. Failing rules it out as a stable cause, and that is the direction
carrying information here.

**Two stages, two different nulls.** Conflating them is the easy mistake.

1. *Association.* Statistic: pooled rank AUC, ``z = |A - 0.5| / SE``. Null: the
   labels carry no information, so **y is permuted**. Screens all surviving
   sensors; BH at ``--alpha`` across them.

2. *Invariance*, asked only of sensors that survived stage 1 -- an unassociated
   sensor is invariant for the uninteresting reason. Statistic: Cochran's Q of
   the within-block AUCs against their inverse-variance mean. Null: the
   association is *the same in every block*, which is emphatically not "there is
   no association". So **the block assignment is permuted**, not the labels:
   rows are reshuffled into blocks of the same sizes, which preserves each
   sensor's overall association exactly while destroying any block-specific
   structure. Permuting y here would build the reference distribution at
   ``A = 0.5``, where AUC has smaller sampling variance than at the observed
   ``A``, and would therefore declare strongly-associated sensors non-invariant
   for no reason but their strength. BH within stage 1's survivors.

**Why permutation and not the closed form.** The Hanley-McNeil SE is an
approximation and SECOM's sensors carry heavy ties, so Cochran's Q does not
follow its nominal chi^2 here: measured against the null, the chi^2 test rejects
well above its nominal rate (reported as `chi2_calibration`, a diagnostic, not a
result). Every decision below uses ``p = (1 + #{stat_null >= stat_obs}) / (B+1)``,
which is correctly sized for each sensor whatever its ties, missingness or
per-block positive counts. The cost is resolution: no p-value can fall below
``1/(B+1)``, so ``--assoc-perm`` must be large enough that BH can reject at all.

    OMP_NUM_THREADS=1 python scripts/invariance.py --blocks 5 \
        --assoc-perm 20000 --inv-perm 20000
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import sklearn
from scipy import stats

from yieldrca.data import load_secom
from yieldrca.preprocess import SensorCleaner


# ---------------------------------------------------------------- statistics
def rank_table(X):
    """Within-column ranks, missing entries left as NaN and excluded from the
    ranking of the values that are present."""
    return stats.rankdata(X, axis=0, nan_policy="omit")


def auc_from_ranks(R, mask, ypos):
    """Rank AUC and Hanley-McNeil SE per column, over observed rows only.

    ``R`` are within-column ranks among observed values, ``mask`` marks observed
    entries and ``ypos`` is the boolean positive-class indicator for the rows of
    ``R``. Sensors with no observed positive or no observed negative get NaN.
    """
    n1 = mask[ypos].sum(axis=0).astype(float)
    n0 = mask[~ypos].sum(axis=0).astype(float)
    srank = np.where(mask[ypos], R[ypos], 0.0).sum(axis=0)
    ok = (n1 >= 1) & (n0 >= 1)
    a = np.full(R.shape[1], np.nan)
    a[ok] = (srank[ok] - n1[ok] * (n1[ok] + 1) / 2.0) / (n0[ok] * n1[ok])
    with np.errstate(invalid="ignore"):
        q1 = a / (2.0 - a)
        q2 = 2.0 * a * a / (1.0 + a)
        var = (a * (1 - a) + (n1 - 1) * (q1 - a * a)
               + (n0 - 1) * (q2 - a * a)) / (n0 * n1)
    return a, np.sqrt(np.maximum(var, 1e-12))


def cochran_q(A, S):
    """Cochran's Q per sensor from ``(n_sensors, n_blocks)`` AUC and SE tables."""
    ok = np.isfinite(A) & np.isfinite(S) & (S > 0)
    W = np.where(ok, 1.0 / np.where(ok, S, 1.0) ** 2, 0.0)
    sw = W.sum(axis=1)
    Az = np.where(ok, A, 0.0)
    mean = np.where(sw > 0, (W * Az).sum(axis=1) / np.where(sw > 0, sw, 1.0), np.nan)
    q = (W * (Az - mean[:, None]) ** 2 * ok).sum(axis=1)
    df = ok.sum(axis=1) - 1
    return np.where(df >= 1, q, np.nan), mean, df


def bh(p):
    """Benjamini-Hochberg adjusted p-values; non-finite entries pass as 1.0."""
    p = np.asarray(p, dtype=float)
    out = np.ones_like(p)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok) == 0:
        return out
    ps = p[ok]
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        prev = min(prev, ps[order[rank]] * m / (rank + 1))
        adj[order[rank]] = prev
    out[ok] = np.minimum(adj, 1.0)
    return out


# ------------------------------------------------------------------- stage 1
def association_stage(X, y, n_perm, seed=0, report_every=5000):
    """Pooled rank association per sensor, with a label-permutation p-value.

    Ranks do not depend on the labels, so the rank table is built once and each
    permutation costs one masked column-sum.
    """
    mask = np.isfinite(X)
    R = rank_table(X)
    R = np.where(np.isfinite(R), R, 0.0)
    ypos = y.astype(bool)
    a_obs, se_obs = auc_from_ranks(R, mask, ypos)
    with np.errstate(invalid="ignore"):
        z_obs = np.abs(a_obs - 0.5) / np.where(se_obs > 0, se_obs, np.nan)

    ge = np.zeros(X.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    for b in range(n_perm):
        yy = rng.permutation(ypos)
        a, se = auc_from_ranks(R, mask, yy)
        with np.errstate(invalid="ignore"):
            z = np.abs(a - 0.5) / np.where(se > 0, se, np.nan)
        ge += np.where(np.isfinite(z) & np.isfinite(z_obs), z >= z_obs, False)
        if report_every and (b + 1) % report_every == 0:
            print(f"  [assoc] {b + 1}/{n_perm}", flush=True)
    p = np.where(np.isfinite(z_obs), (1.0 + ge) / (n_perm + 1.0), np.nan)
    return {"auc": a_obs, "se": se_obs, "z": z_obs, "p": p, "n_perm": n_perm}


# ------------------------------------------------------------------- stage 2
def _block_stats(Xsub, y, blocks):
    """(A, S) tables for one assignment of rows to blocks."""
    E = len(blocks)
    A = np.full((Xsub.shape[1], E), np.nan)
    S = np.full((Xsub.shape[1], E), np.nan)
    for e, rows in enumerate(blocks):
        Xb = Xsub[rows]
        m = np.isfinite(Xb)
        R = rank_table(Xb)
        R = np.where(np.isfinite(R), R, 0.0)
        A[:, e], S[:, e] = auc_from_ranks(R, m, y[rows].astype(bool))
    return A, S


def invariance_stage(Xsub, y, blocks, n_perm, seed=0, report_every=5000):
    """Cochran's Q across blocks with a *block-assignment* permutation null.

    Sizes are held fixed and rows are reshuffled between blocks, so each
    permutation preserves every sensor's pooled association exactly and varies
    only which wafers count as which production period -- the null of "the same
    association everywhere".
    """
    A_obs, S_obs = _block_stats(Xsub, y, blocks)
    q_obs, iv_mean, df = cochran_q(A_obs, S_obs)
    sizes = [len(b) for b in blocks]
    n = Xsub.shape[0]

    ge = np.zeros(Xsub.shape[1], dtype=np.int64)
    chi2_rej = 0
    n_valid = int(np.isfinite(q_obs).sum())
    rng = np.random.default_rng(seed)
    for b in range(n_perm):
        perm = rng.permutation(n)
        pb, off = [], 0
        for s in sizes:
            pb.append(perm[off:off + s])
            off += s
        A, S = _block_stats(Xsub, y, pb)
        qn, _, dfn = cochran_q(A, S)
        ge += np.where(np.isfinite(qn) & np.isfinite(q_obs), qn >= q_obs, False)
        with np.errstate(invalid="ignore"):
            pn = stats.chi2.sf(qn, np.maximum(dfn, 1))
        chi2_rej += int(np.nansum(pn < 0.05))
        if report_every and (b + 1) % report_every == 0:
            print(f"  [inv] {b + 1}/{n_perm}", flush=True)

    p = np.where(np.isfinite(q_obs), (1.0 + ge) / (n_perm + 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        i2 = np.where(q_obs > 0, np.maximum(0.0, (q_obs - df) / np.where(q_obs > 0, q_obs, 1.0)), np.nan)
    return {"A": A_obs, "S": S_obs, "q": q_obs, "p": p, "i2": i2,
            "iv_mean": iv_mean, "n_perm": n_perm,
            "chi2_size": float(chi2_rej / max(1, n_perm * max(n_valid, 1)))}


# -------------------------------------------------------------- power audit
def power_ladder(y, blocks, deltas, n_rep, n_perm, alpha, n_family, seed=7):
    """How large must a break in association be before this test sees it?

    "Nothing was non-invariant" is only evidence of invariance if the test could
    have found non-invariance. So sensors are *built* with a known break --
    within block 0 the positives are shifted so that block's AUC is
    ``0.5 + delta`` while every other block sits at 0.5 -- and the same
    permutation test is run on them. The detection rate is the power at that
    effect size, at this dataset's actual block sizes and fail counts.

    Two thresholds are reported: the unadjusted ``alpha``, and
    ``alpha / n_family``, the strictest level Benjamini-Hochberg could have
    demanded of the real test given how many sensors entered it. The truth for
    the real result sits between them.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    ypos = y.astype(bool)
    b0 = blocks[0]
    out = []
    for d in deltas:
        target = 0.5 + d
        shift = np.sqrt(2.0) * stats.norm.ppf(min(max(target, 0.501), 0.999))
        cols = rng.standard_normal((n, n_rep))
        # break the association inside block 0 only
        rows = b0[ypos[b0]]
        cols[rows] += shift
        res = invariance_stage(cols, y, blocks, n_perm, seed=int(rng.integers(1 << 30)),
                               report_every=0)
        pv = res["p"]
        out.append({
            "delta": float(d),
            "target_block0_auc": float(target),
            "shift": float(shift),
            "n_replicates": int(n_rep),
            "median_p": float(np.nanmedian(pv)),
            "power_at_alpha": float(np.nanmean(pv < alpha)),
            "power_at_alpha_over_family": float(np.nanmean(pv < alpha / max(n_family, 1))),
            "median_i2": float(np.nanmedian(res["i2"])),
        })
        print(f"  [power] block-0 AUC {target:.2f}: detected "
              f"{out[-1]['power_at_alpha']:.0%} at p<{alpha}, "
              f"{out[-1]['power_at_alpha_over_family']:.0%} at "
              f"p<{alpha}/{n_family}", flush=True)
    return out


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--assoc-perm", type=int, default=20000)
    ap.add_argument("--inv-perm", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power-rep", type=int, default=200,
                    help="synthetic sensors per effect size in the power audit")
    ap.add_argument("--power-perm", type=int, default=2000)
    ap.add_argument("--root", default="data")
    ap.add_argument("--eval", default="runs/secom_eval.json")
    ap.add_argument("--out", default="runs/invariance.json")
    a = ap.parse_args()

    X, y, names, t = load_secom(a.root, with_time=True)
    cleaner = SensorCleaner().fit(X)
    Xc = cleaner.transform(X)
    keep = cleaner.keep_
    order = np.argsort(t, kind="stable")
    blocks = np.array_split(order, a.blocks)
    print(f"[invariance] {Xc.shape[1]} sensors survive cleaning, "
          f"{a.blocks} time blocks", flush=True)
    block_desc = [{"block": e, "n": int(len(idx)), "n_fail": int(y[idx].sum()),
                   "fail_rate": float(y[idx].mean())} for e, idx in enumerate(blocks)]
    for b in block_desc:
        print(f"  block {b['block']}: n={b['n']:4d}  fails={b['n_fail']:3d}  "
              f"rate={b['fail_rate']:.3f}", flush=True)

    # ---- what the agent loop actually reported, as already measured --------
    ev = json.loads(Path(a.eval).read_text())
    agent_recs = [r for r in ev["records"] if r["arm"] == "agent_rf"]
    n_folds = len(agent_recs)
    sel_count = Counter(int(j) for r in agent_recs for j in r["selected"])
    top5_count = Counter(int(j) for r in agent_recs for j in r["top5"])
    o2c = {int(o): i for i, o in enumerate(keep)}
    gi = lambda lst: sorted({o2c[j] for j in lst if j in o2c})
    suspects_top5 = gi([j for j, _ in top5_count.most_common(5)])
    suspects_major = gi([j for j, c in sel_count.items() if c >= n_folds / 2])
    suspects_any = gi(list(sel_count))

    # ---- stage 1: association ---------------------------------------------
    print(f"[invariance] stage 1: association, {a.assoc_perm} label permutations",
          flush=True)
    assoc = association_stage(Xc, y, a.assoc_perm, seed=0)
    assoc_bh = bh(assoc["p"])
    associated = assoc_bh < a.alpha
    assoc_idx = np.flatnonzero(associated)
    floor = 1.0 / (a.assoc_perm + 1.0)
    print(f"[invariance] {len(assoc_idx)} of {Xc.shape[1]} sensors associated "
          f"(BH {a.alpha}); permutation resolution floor {floor:.2e}", flush=True)

    # ---- stage 2: invariance, asked only of the associated -----------------
    tested = assoc_idx
    if len(tested) == 0:
        inv = None
        invariant = np.zeros(Xc.shape[1], dtype=bool)
        non_invariant = np.zeros(Xc.shape[1], dtype=bool)
        inv_bh_full = np.full(Xc.shape[1], np.nan)
        i2_full = np.full(Xc.shape[1], np.nan)
        A_full = np.full((Xc.shape[1], a.blocks), np.nan)
    else:
        print(f"[invariance] stage 2: invariance of those {len(tested)}, "
              f"{a.inv_perm} block-assignment permutations", flush=True)
        inv = invariance_stage(Xc[:, tested], y, blocks, a.inv_perm, seed=1)
        inv_bh = bh(inv["p"])
        non_inv_sub = inv_bh < a.alpha
        non_invariant = np.zeros(Xc.shape[1], dtype=bool)
        non_invariant[tested] = non_inv_sub
        invariant = associated & ~non_invariant
        inv_bh_full = np.full(Xc.shape[1], np.nan)
        inv_bh_full[tested] = inv_bh
        i2_full = np.full(Xc.shape[1], np.nan)
        i2_full[tested] = inv["i2"]
        A_full = np.full((Xc.shape[1], a.blocks), np.nan)
        A_full[tested] = inv["A"]
        print(f"[invariance] chi^2 diagnostic: nominal test would reject at "
              f"{inv['chi2_size']:.3f} under the null (nominal 0.050)", flush=True)

    # ---- power: could this test have seen a break if there were one? ------
    n_family = max(len(tested), 1)
    print(f"[invariance] power audit against injected breaks "
          f"({a.power_rep} sensors x {a.power_perm} permutations per level)",
          flush=True)
    power = power_ladder(y, blocks, [0.05, 0.10, 0.15, 0.20, 0.25],
                         a.power_rep, a.power_perm, a.alpha, n_family)

    def describe(idx, label):
        idx = np.asarray(sorted(set(int(i) for i in idx)), dtype=int)
        if len(idx) == 0:
            return {"label": label, "n": 0, "n_associated": 0,
                    "n_non_invariant": 0, "n_invariant": 0, "sensors": []}
        return {
            "label": label,
            "n": int(len(idx)),
            "n_associated": int(associated[idx].sum()),
            "n_non_invariant": int(non_invariant[idx].sum()),
            "n_invariant": int(invariant[idx].sum()),
            "frac_associated": float(associated[idx].mean()),
            "median_i2_of_associated": (float(np.nanmedian(i2_full[idx][associated[idx]]))
                                        if associated[idx].sum() else float("nan")),
            "sensors": [int(keep[i]) for i in idx],
        }

    groups = [
        describe(suspects_top5, "agent loop, consensus top-5"),
        describe(suspects_major, f"agent loop, selected in >=50% of {n_folds} folds"),
        describe(suspects_any, f"agent loop, selected in >=1 of {n_folds} folds"),
        describe(np.setdiff1d(assoc_idx, np.asarray(suspects_any, dtype=int)),
                 "associated but never selected by the loop"),
        describe(np.arange(Xc.shape[1]), "all surviving sensors"),
    ]

    rows = []
    for i in sorted(set(suspects_major) | set(suspects_top5) | set(assoc_idx.tolist())):
        rows.append({
            "sensor": int(keep[i]),
            "name": names[int(keep[i])],
            "folds_selected": int(sel_count.get(int(keep[i]), 0)),
            "folds_top5": int(top5_count.get(int(keep[i]), 0)),
            "pooled_auc": float(assoc["auc"][i]),
            "assoc_p": float(assoc["p"][i]),
            "assoc_p_bh": float(assoc_bh[i]),
            "associated": bool(associated[i]),
            "block_auc": [None if not np.isfinite(v) else float(v) for v in A_full[i]],
            "invariance_p_bh": (None if not np.isfinite(inv_bh_full[i])
                                else float(inv_bh_full[i])),
            "i2": None if not np.isfinite(i2_full[i]) else float(i2_full[i]),
            "invariant": bool(invariant[i]),
        })
    rows.sort(key=lambda r: (-r["folds_selected"], r["assoc_p"]))

    out = {
        "protocol": {
            "environments": f"{a.blocks} contiguous equal-count time blocks in "
                            "timestamp order",
            "stage1_association": "pooled rank AUC, z = |AUC-0.5|/SE "
                                  "(Hanley-McNeil); null = permuted labels; "
                                  f"B = {a.assoc_perm}; BH at {a.alpha} across "
                                  f"all {int(Xc.shape[1])} surviving sensors",
            "stage2_invariance": "Cochran's Q of within-block AUCs about their "
                                 "inverse-variance mean; null = rows reshuffled "
                                 "into same-sized blocks, which holds each "
                                 "sensor's pooled association fixed and destroys "
                                 f"only the block structure; B = {a.inv_perm}; "
                                 f"BH at {a.alpha} within stage 1's survivors",
            "why_two_nulls": "permuting labels would build the invariance "
                             "reference at AUC 0.5, where the statistic has "
                             "smaller sampling variance than at the observed "
                             "AUC, and would flag strong sensors as "
                             "non-invariant for their strength alone",
            "invariant": "associated (stage 1) AND homogeneity not rejected "
                         "(stage 2)",
            "identification": "necessary condition for a stable cause, not "
                              "sufficient; a marginal screen, not full ICP",
            "citation_note": "the ICP framework is Peters, Buhlmann & "
                             "Meinshausen (JRSS-B 2016); no number from that "
                             "paper is used here, only the argument",
            "resolution_floor": floor,
            "alpha": a.alpha, "n_blocks": a.blocks,
            "suspects_from": f"{a.eval}, arm agent_rf, {n_folds} CV folds",
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
        },
        "blocks": block_desc,
        "chi2_calibration": {
            "chi2_rejection_rate_under_null": (None if inv is None else inv["chi2_size"]),
            "nominal": 0.05,
            "note": "share of (sensor, permutation) draws where the *nominal* "
                    "chi^2 heterogeneity test rejects at 0.05 although the null "
                    "holds. Above 0.05 means the closed form is "
                    "anticonservative on this data, which is why every decision "
                    "here uses the permutation p-value. Diagnostic, not result.",
        },
        "power": {
            "design": "synthetic sensors whose association is 0.5 + delta in "
                      "block 0 and 0.5 in every other block, run through the "
                      "identical stage-2 test at this dataset's real block "
                      "sizes and fail counts",
            "n_family": int(n_family),
            "note": "a null result in `groups` means 'no break this test can "
                    "see', and this table says how big a break that is",
            "ladder": power,
        },
        "totals": {
            "n_sensors": int(Xc.shape[1]),
            "n_associated": int(associated.sum()),
            "n_tested_for_invariance": int(len(tested)),
            "n_non_invariant": int(non_invariant.sum()),
            "n_invariant": int(invariant.sum()),
            "n_ever_selected_by_loop": int(len(suspects_any)),
            "n_loop_selected_and_associated": int(associated[np.asarray(suspects_any, dtype=int)].sum())
            if suspects_any else 0,
        },
        "groups": groups,
        "per_sensor": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\nof {Xc.shape[1]} sensors: {int(associated.sum())} associated, "
          f"{int(non_invariant.sum())} of those non-invariant across periods, "
          f"{int(invariant.sum())} associated AND invariant")
    print(f"\n{'group':<50}{'n':>4}{'assoc':>7}{'inv':>6}")
    for g in groups:
        if g.get("n"):
            print(f"{g['label']:<50}{g['n']:>4}{g['n_associated']:>7}{g['n_invariant']:>6}")
    print(f"\n{'injected block-0 AUC':<24}{'power @.05':>12}{'power @BH-worst':>17}")
    for row in power:
        print(f"{row['target_block0_auc']:<24.2f}{row['power_at_alpha']:>12.0%}"
              f"{row['power_at_alpha_over_family']:>17.0%}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
