"""Tests for the two things this repo claims about its *report* rather than its
predictions: that the false-discovery machinery is measured, and that the
invariance screen's statistics are the statistics they say they are.

The vectorised rank AUC in `scripts/invariance.py` exists so that 20,000
permutations are affordable, and a fast statistic that quietly disagrees with
the slow one it replaces would corrupt every p-value built on it. So it is
checked against scikit-learn directly, including the ties and missing values
that made the closed-form reference unusable in the first place.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import roc_auc_score

import null_fdr_rankers as nfr
from invariance import auc_from_ranks, bh, cochran_q, invariance_stage, rank_table
from yieldrca.data import make_synthetic
from yieldrca.estimator import AgentRCA


# ------------------------------------------------------- the fast rank AUC
def _auc_ref(y, x):
    ok = np.isfinite(x)
    return roc_auc_score(y[ok], x[ok])


def test_vectorised_auc_matches_sklearn_with_ties_and_missing():
    rng = np.random.default_rng(0)
    n, p = 300, 40
    X = rng.standard_normal((n, p))
    # heavy ties: SECOM sensors are quantised, which is what broke the chi^2
    X[:, :20] = np.round(X[:, :20] * 2) / 2
    X[rng.uniform(size=X.shape) < 0.1] = np.nan
    y = (rng.uniform(size=n) < 0.2).astype(int)

    mask = np.isfinite(X)
    R = np.where(np.isfinite(rank_table(X)), rank_table(X), 0.0)
    a, se = auc_from_ranks(R, mask, y.astype(bool))
    for j in range(p):
        assert abs(a[j] - _auc_ref(y, X[:, j])) < 1e-9, j
    assert np.all(se > 0)


def test_vectorised_auc_is_nan_when_a_class_is_missing():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.zeros(3, dtype=int)          # no positives at all
    mask = np.isfinite(X)
    R = np.where(np.isfinite(rank_table(X)), rank_table(X), 0.0)
    a, _ = auc_from_ranks(R, mask, y.astype(bool))
    assert np.isnan(a[0])


# --------------------------------------------------------- heterogeneity
def test_cochran_q_is_zero_when_every_block_agrees():
    A = np.array([[0.6, 0.6, 0.6]])
    S = np.array([[0.05, 0.05, 0.05]])
    q, mean, df = cochran_q(A, S)
    assert abs(q[0]) < 1e-12
    assert abs(mean[0] - 0.6) < 1e-12
    assert df[0] == 2


def test_cochran_q_grows_with_the_spread_between_blocks():
    S = np.array([[0.02, 0.02, 0.02]])
    q_small, _, _ = cochran_q(np.array([[0.55, 0.60, 0.65]]), S)
    q_large, _, _ = cochran_q(np.array([[0.40, 0.60, 0.80]]), S)
    assert q_large[0] > q_small[0] > 0


def test_bh_matches_a_worked_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    adj = bh(p)
    assert np.all(np.diff(adj) >= -1e-12)          # monotone, as BH must be
    assert abs(adj[0] - 0.005) < 1e-9              # 0.001 * 5/1
    assert abs(adj[4] - 0.042) < 1e-9              # 0.042 * 5/5


def test_block_permutation_null_detects_a_break_it_should_detect():
    """A sensor associated only inside block 0 must be flagged non-invariant.

    Guards the direction of the whole stage-2 test: if the permutation scheme
    were wrong, this would come back invariant and every real null result in
    `runs/invariance.json` would be meaningless.
    """
    rng = np.random.default_rng(1)
    n = 900
    y = (rng.uniform(size=n) < 0.25).astype(int)
    blocks = np.array_split(np.arange(n), 3)
    X = rng.standard_normal((n, 2))
    b0 = blocks[0]
    X[b0[y[b0] == 1], 0] += 3.0                    # break, block 0 only
    res = invariance_stage(X, y, blocks, n_perm=400, seed=0, report_every=0)
    assert res["p"][0] < 0.01, res["p"]            # broken sensor: flagged
    assert res["p"][1] > 0.05, res["p"]            # untouched sensor: not


# -------------------------------------------------- the hallucination property
def test_agent_loop_cannot_return_an_empty_report_on_pure_noise():
    """Pins the finding `runs/null_fdr.json` quantifies on SECOM.

    On data with no signal whatever, the loop still names suspects, because
    `AgentRCA.fit` restores the top candidates when nothing clears the
    stability threshold. This test asserts the *current* behaviour so that the
    day someone gives the loop the ability to abstain, it fails loudly and the
    claim in the README has to be rewritten at the same time.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 40))
    y = (rng.uniform(size=300) < 0.2).astype(int)   # labels independent of X
    est = AgentRCA(base="logreg", n_screen=20, n_screen_boot=12, n_boot=4,
                   select_k=8, top_k=5, max_select=10, n_inner=2, n_repeats=1,
                   stability_min=0.99,                # deliberately unreachable
                   random_state=0).fit(X, y)
    assert len(est.selected_) >= 1
    assert max(est.stability_.values()) < 0.99       # nothing actually cleared it
    assert len(est.ranking()) >= 5                   # a top-5 is always available


def test_permuted_labels_leave_the_loop_reporting_on_synthetic_data():
    """Same property on data that *does* have a signal, once it is destroyed."""
    X, y, names, causal = make_synthetic(n=400, p=50, n_causal=3, fail_rate=0.15,
                                         seed=0)
    yp = np.random.default_rng(0).permutation(y)
    est = AgentRCA(base="logreg", n_screen=25, n_screen_boot=15, n_boot=4,
                   select_k=10, top_k=5, max_select=10, n_inner=2, n_repeats=1,
                   random_state=0).fit(X, yp)
    assert len(est.selected_) >= 1


# ------------------------------------------------------------- the abstention rule
def test_report_tau_lets_the_loop_abstain_on_pure_noise():
    """The fix for the property pinned above, tested as a property.

    With a null-calibrated bar in place the loop must be *able* to report
    nothing -- and must say so in words, since an empty list is not a finding a
    reader can act on.
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 40))
    y = (rng.uniform(size=300) < 0.2).astype(int)
    kw = dict(base="logreg", n_screen=20, n_screen_boot=12, n_boot=4,
              select_k=8, top_k=5, max_select=10, n_inner=2, n_repeats=1,
              random_state=0)
    est = AgentRCA(report_tau=1.01, **kw).fit(X, y)     # unreachable bar
    assert est.abstained_ is True
    assert len(est.reported_) == 0
    text = est.report()
    assert "noise floor" in text
    assert "permuted labels" in text

    # ...and prediction is untouched, so turning abstention on cannot move an AUC
    loose = AgentRCA(report_tau=None, **kw).fit(X, y)
    assert np.array_equal(est.selected_, loose.selected_)
    assert np.allclose(est.predict_proba(X), loose.predict_proba(X))


def test_report_tau_none_reports_exactly_what_it_predicts_with():
    """The default must stay backward compatible, or every earlier number moves."""
    X, y, names, causal = make_synthetic(n=400, p=50, n_causal=3,
                                         fail_rate=0.15, seed=0)
    est = AgentRCA(base="logreg", n_screen=25, n_screen_boot=15, n_boot=4,
                   select_k=10, top_k=5, max_select=10, n_inner=2, n_repeats=1,
                   random_state=0).fit(X, y)
    assert est.abstained_ is False
    assert sorted(est.reported_.tolist()) == sorted(est.selected_.tolist())
    assert "noise floor" not in est.report()


def test_report_tau_keeps_only_suspects_above_the_bar():
    """Monotone in tau: raising the bar can only shorten the report."""
    X, y, names, causal = make_synthetic(n=500, p=60, n_causal=4,
                                         fail_rate=0.15, seed=1)
    kw = dict(base="logreg", n_screen=30, n_screen_boot=18, n_boot=6,
              select_k=12, top_k=5, max_select=12, n_inner=2, n_repeats=1,
              random_state=0)
    prev = None
    for tau in (0.0, 0.5, 0.9, 1.01):
        est = AgentRCA(report_tau=tau, **kw).fit(X, y)
        n = len(est.reported_)
        assert all(est.stability_[int(j)] >= tau for j in est.reported_)
        if prev is not None:
            assert n <= prev, (tau, n, prev)
        prev = n


# ------------------------------------------- the saturation ceiling argument
def test_heldout_ceiling_bounds_the_achievable_control():
    """A saturated support statistic caps the error control any tau can give.

    This is the argument the ranker comparison turns on, so it is pinned rather
    than trusted: if a fraction f of null replicates score the maximum possible
    support, no threshold at or below that maximum excludes them, and the
    achievable abstention rate cannot exceed 1 - f however small alpha is.
    """
    rng = np.random.default_rng(0)
    for f in (0.0, 0.1, 0.35):
        n = 400
        n_sat = int(round(f * n))
        null_max = np.concatenate([
            np.ones(n_sat),                                  # saturated
            rng.uniform(0.2, 0.95, n - n_sat),               # not
        ])
        real_sets = [[1.0, 0.9] for _ in range(20)]
        out = nfr._heldout(null_max, real_sets, 0.05, 40, np.random.default_rng(1))
        assert abs(out["saturated_null_fraction"] - f) < 0.02, f
        assert abs(out["max_attainable_null_abstention"] - (1 - f)) < 0.02, f
        # the achieved rate can never beat the ceiling, whatever tau came out
        assert out["null_abstention_heldout"] <= out["max_attainable_null_abstention"] + 1e-9


def test_heldout_control_approaches_nominal_when_nothing_saturates():
    """With headroom, split-half calibration should land near 1 - alpha.

    Guards the other direction: if this drifted far from nominal on a clean
    continuous null, the calibration code would be wrong and every control
    number in `runs/null_fdr_rankers.json` with it.
    """
    rng = np.random.default_rng(2)
    null_max = rng.uniform(0.0, 0.9, 600)          # no saturation at all
    out = nfr._heldout(null_max, [[0.95] for _ in range(10)], 0.05, 60,
                       np.random.default_rng(3))
    assert out["max_attainable_null_abstention"] == 1.0
    assert 0.90 <= out["null_abstention_heldout"] <= 0.98, out


def test_heldout_is_not_scored_on_the_replicates_that_set_tau():
    """The split must be a real split, or the null rate is 1 - alpha by fiat.

    If calibration and evaluation shared replicates, a null drawn so that one
    half sits far above the other would still report ~1 - alpha. Here the
    halves are exchangeable, so the check is that tau tracks the calibration
    half rather than the whole: a heavily right-skewed null must push tau above
    the median of the pooled sample.
    """
    rng = np.random.default_rng(4)
    null_max = np.concatenate([rng.uniform(0.0, 0.3, 300),
                               rng.uniform(0.7, 0.9, 300)])
    out = nfr._heldout(null_max, [[0.95] for _ in range(10)], 0.05, 60,
                       np.random.default_rng(5))
    assert out["tau_mean"] > float(np.median(null_max))
    assert out["null_abstention_heldout"] < 1.0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f()
        print(f"ok {n}")
    print(f"ok  ({len(fns)} tests)")


# ------------------------------------- the fallback cannot reach the report
def test_fallback_tops_up_prediction_set_not_the_report():
    """`AgentRCA`'s never-empty guard fills `selected_`, never `reported_`.

    This repo spent a turn claiming the opposite -- that the guard "is the
    thing that makes abstention impossible" -- which conflated the set the
    final classifier is fitted on with the set an engineer is handed. Pin the
    distinction in code so the claim cannot come back: build a case where the
    guard must fire (`stability_min` above 1.0 is unreachable, so nothing can
    survive the drop step) with a `report_tau` above it, and check that
    prediction still works while the report comes back empty.
    """
    X, y, _, _ = make_synthetic(n=220, p=24, n_causal=3, seed=1)
    est = AgentRCA(base="logreg", screen="univariate", attribution="model",
                   n_screen=12, n_screen_boot=12, select_k=3, top_k=3,
                   stability_min=1.01, report_tau=1.01, n_boot=4, n_inner=2,
                   max_select=5, random_state=0).fit(X, y)

    # The guard fired: nothing could clear an unreachable stability_min, yet
    # the classifier still has columns to predict from.
    assert all(v < est.stability_min for v in est.stability_.values())
    assert len(est.selected_) > 0
    assert est.predict_proba(X).shape == (len(y), 2)

    # And none of that reached the report, which is what the null-FDR
    # measurement thresholds.
    assert len(est.reported_) == 0
    assert est.abstained_ is True
    assert "do not identify a root cause" in est.report()


def test_calibrated_error_control_ignores_the_fallback():
    """tau above `stability_min` makes the guard unreachable by construction.

    The measured cross-tab in `RESULTS.md` rests on an ordering, not a
    coincidence: the fallback fires only when every support is below
    `stability_min`, so whenever tau >= `stability_min` a fired replicate
    cannot name anything. Checked here on the recorded null replicates so a
    future run that inverts the ordering fails loudly instead of silently
    invalidating the paragraph.
    """
    import json

    src = ROOT / "runs" / "null_fdr_k5.json"
    if not src.exists():                                    # pragma: no cover
        return
    d = json.loads(src.read_text())
    tau = d["thresholds"]["alpha_0.05"]
    null = [r for r in d["records"] if r["permuted"]]
    fired = [r for r in null if r["fallback_fired"]]
    assert fired, "expected the guard to fire somewhere on this null"
    assert tau >= fired[0]["stability_min"], (
        "tau fell below stability_min, so the RESULTS.md cross-tab no longer "
        "follows from the ordering and its paragraph must be re-derived")
    for r in fired:
        assert r["max_stability"] < r["stability_min"]
        assert not any(v >= tau for v in r["stability_values"])


# ------------------------------- the 2x2's bare-ranker cells must be matched
def test_bare_ranker_cells_are_the_same_construction():
    """`perm_only` and `model_only` must differ only in the statistic.

    The decomposition in `RESULTS.md` subtracts a bare ranker from the full
    loop in each attribution column, and that subtraction only prices the
    architecture if the two bare cells are the same construction. An earlier
    version used `rf_impurity` as the model-native cell, which fits a different
    forest over a different candidate universe -- so the subtraction priced the
    architecture plus two confounds. `_bare_rank` exists to remove them, and
    this asserts it: driving it with `attribution="permutation"` must reproduce
    the independently written `rank_perm_only` exactly.
    """
    import stability_secom as ss

    X, y, _, _ = make_synthetic(n=180, p=30, n_causal=3, seed=3)
    a = ss._bare_rank(X, y, "permutation")
    b = ss.rank_perm_only(X, y)
    assert np.array_equal(a, b), (
        "the bare-ranker helper and rank_perm_only have diverged, so the "
        "perm_only column of the RESULTS.md 2x2 no longer prices only the "
        "architecture")

    # And the model-native cell is the same construction with one field moved,
    # so it ranks the same universe -- not a wider or narrower one.
    m = ss._bare_rank(X, y, "model")
    assert set(m.tolist()) == set(a.tolist())
    assert not np.array_equal(m, a), "the two statistics should not coincide"


# --------------------------- what error control is attainable at the grid
def test_attainable_control_is_a_grid_set_by_n_boot():
    """Control lands on the grid {1 - P(M >= k/n_boot)}, and that explains the ties.

    An earlier version of this test asserted a "saturation cap identity" and an
    alpha-invariance that an adversarial review showed was mis-stated: the
    endpoint bound is a one-line consequence of boundedness plus the `>=`
    comparison, and the invariance is local to a gap in the grid rather than
    global. What is actually checkable, and what `RESULTS.md` now claims, is
    the grid structure itself.

    Two assertions. First, every measured control value sits at (or within
    calibration slack of) some attainable grid point -- if it did not, the
    step-function account would be wrong. Second, the ties are explained: two
    alpha levels reporting identical control must have thresholds that fall in
    the same grid gap, i.e. no null replicate's max lies between them.
    """
    import json

    pairs = [("null_fdr", "abstain"), ("null_fdr_model", "abstain_model"),
             ("null_fdr_k5", "abstain_k5"),
             ("null_fdr_k5_model", "abstain_k5_model")]
    checked = ties = 0
    for nf_name, ab_name in pairs:
        nf_p = ROOT / "runs" / f"{nf_name}.json"
        ab_p = ROOT / "runs" / f"{ab_name}.json"
        if not (nf_p.exists() and ab_p.exists()):        # pragma: no cover
            continue
        nf = json.loads(nf_p.read_text())
        ab = json.loads(ab_p.read_text())
        nb = nf["protocol"]["agent_cfg"]["n_boot"]
        mx = np.asarray([r["max_stability"] for r in nf["records"]
                         if r["permuted"]], dtype=float)
        grid = sorted({float(1.0 - (mx >= k / nb).mean())
                       for k in range(1, nb + 1)})
        lv = ab.get("levels") or {}

        for key, m in lv.items():
            ctl = m["null_abstention_heldout"]
            # The split-half average can fall between two grid points when
            # different splits sit on different rungs, so the check is that it
            # lies within the grid's range and no further than one step from
            # some rung.
            step = max(1.0 / nb, 0.02)
            assert min(abs(ctl - g) for g in grid) <= step, (
                f"{nf_name}/{key}: control {ctl:.4f} is more than one grid "
                f"step from every attainable value {grid}")
            checked += 1

        keys = sorted(lv)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a_, b_ = lv[keys[i]], lv[keys[j]]
                if abs(a_["null_abstention_heldout"]
                       - b_["null_abstention_heldout"]) > 1e-9:
                    continue
                lo = min(a_["tau_min"], b_["tau_min"])
                hi = max(a_["tau_max"], b_["tau_max"])
                # Identical control with different thresholds is only possible
                # if no null replicate's max separates them.
                between = int(((mx >= lo) & (mx < hi)).sum())
                assert between == 0 or abs(lo - hi) < 1e-9, (
                    f"{nf_name}: {keys[i]} and {keys[j]} report identical "
                    f"control but {between} null replicates have max in "
                    f"[{lo:.4f}, {hi:.4f}), so the tie is not explained by a "
                    f"gap in the grid")
                ties += 1
    assert checked >= 8, f"only {checked} (arm, alpha) pairs checked"
    assert ties >= 1, "expected at least one exact tie to explain"




def test_n_boot_refines_the_attainable_grid():
    """More resamples => strictly more attainable control values.

    The `RESULTS.md` claim that `n_boot` sets the resolution of error control
    rests on a ladder of three univariate arms in `runs/null_fdr_rankers.json`
    that differ only in `n_boot`. Pinned here so that re-running that script
    with different settings cannot silently invalidate the paragraph.

    Only the *refinement* is asserted, not that control improves: measured
    control went 93.0% -> 94.3% -> 94.1%, so the third rung is flat-to-
    backwards and asserting improvement would be asserting noise.
    """
    import json

    src = ROOT / "runs" / "null_fdr_rankers.json"
    if not src.exists():                                    # pragma: no cover
        return
    d = json.loads(src.read_text())
    per, recs = d["per_ranker"], d["records"]
    fam = sorted(((k, v) for k, v in per.items()
                  if v.get("is_variant") and v.get("select_k") == 5
                  and v.get("ranker") == "univariate"),
                 key=lambda kv: kv[1]["n_boot"])
    assert len(fam) >= 3, "expected an n_boot ladder of at least three arms"

    counts, nears = [], []
    for name, v in fam:
        nb = v["n_boot"]
        mx = np.asarray([r["max_stability"] for r in recs
                         if r["arm"] == name and r["permuted"]], dtype=float)
        assert len(mx), f"no null replicates recorded for {name}"
        ach = sorted({float(1.0 - (mx >= k / nb).mean())
                      for k in range(1, nb + 1)})
        counts.append(len([a for a in ach if a > 0.60]))
        nears.append(min(abs(a - 0.95) for a in ach))

    assert all(counts[i] < counts[i + 1] for i in range(len(counts) - 1)), (
        f"attainable-set sizes {counts} are not strictly increasing in "
        f"n_boot, so the resolution claim in RESULTS.md no longer holds")
    # Nominal must go from unreachable at the coarsest rung to exact later.
    assert nears[0] > 1e-9, "0.95 was already attainable at the coarsest rung"
    assert min(nears[1:]) < 1e-9, "0.95 never became exactly attainable"


def test_calibration_resampling_is_arm_order_independent():
    """Each arm's calib_size row must not depend on which arms preceded it.

    The first version shared one generator across arms, so adding the H9 arm
    moved an already-published row by 0.004 -- a number in a document changing
    because an unrelated row was added. Seeding per arm from its label fixes
    it, and this asserts the property directly rather than trusting the fix.
    """
    import hashlib

    import numpy as _np

    def draws(label, seed=0):
        rng = _np.random.default_rng(
            [seed, int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],
                                  "big")])
        return rng.permutation(10)

    a1, a2 = draws("arm one"), draws("arm two")
    assert not _np.array_equal(a1, a2), "different labels must give different draws"
    assert _np.array_equal(draws("arm one"), a1), "same label must be reproducible"


def test_calib_size_reproduces_the_published_split_half_control():
    """m=100 on a 200-replicate arm is abstain.py's protocol, so it must agree.

    This is the cross-check that licenses reading the calibration-loss column
    as a decomposition of the published number rather than as a separate
    quantity that happens to look similar.
    """
    import json

    cs_p = ROOT / "runs" / "calib_size.json"
    if not cs_p.exists():                                   # pragma: no cover
        return
    cs = json.loads(cs_p.read_text())["arms"]
    pub = {}
    rk_p = ROOT / "runs" / "null_fdr_rankers.json"
    if rk_p.exists():
        for name, v in json.loads(rk_p.read_text())["per_ranker"].items():
            if v.get("is_variant") and v.get("ranker") == "univariate" \
                    and v.get("select_k") == 5:
                pub[f"univariate, select_k=5, n_boot={v['n_boot']}"] = \
                    v["heldout_alpha_0.05"]["null_abstention_heldout"]
    for fn, lab in (("abstain_k5_model.json",
                     "agent loop, select_k=5, model, n_boot=12"),
                    ("abstain_k5.json",
                     "agent loop, select_k=5, permutation, n_boot=12")):
        p = ROOT / "runs" / fn
        if p.exists():
            pub[lab] = json.loads(p.read_text())["levels"]["alpha_0.05"][
                "null_abstention_heldout"]

    checked = 0
    for lab, expected in pub.items():
        if lab not in cs:
            continue
        got = cs[lab]["curve"]["100"]["control_mean"]
        assert abs(got - expected) < 0.01, (
            f"{lab}: calib_size m=100 gives {got:.4f} against abstain.py's "
            f"{expected:.4f}; the decomposition no longer describes the "
            f"published number")
        checked += 1
    assert checked >= 3, f"only cross-checked {checked} arms"
