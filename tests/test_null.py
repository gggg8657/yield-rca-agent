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


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f()
        print(f"ok {n}")
    print(f"ok  ({len(fns)} tests)")
