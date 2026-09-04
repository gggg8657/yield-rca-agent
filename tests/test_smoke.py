"""Smoke tests for the dependency-free path: numpy only, no downloads."""
import numpy as np

from yieldrca import make_synthetic, run_rca
from yieldrca.pipeline import cv_auc, stratified_folds
from yieldrca.model import LogisticRCA, roc_auc


def test_runs_and_recovers():
    X, y, names, causal = make_synthetic(seed=1)
    out = run_rca(X, y, names)
    assert 0.0 <= out["auc"] <= 1.0
    assert out["auc"] > 0.75, f"cross-validated AUC too low: {out['auc']}"
    assert len(out["ranked"]) >= 1
    found = {i for i, _ in out["ranked"]}
    hit = len(found & set(causal.tolist()))
    assert hit >= max(1, len(causal) // 2), f"recovered only {hit}/{len(causal)}"


def test_report_is_markdown():
    X, y, names, _ = make_synthetic(seed=2)
    out = run_rca(X, y, names)
    assert out["report"].startswith("# Yield Root-Cause Report")


def test_cv_auc_is_not_in_sample():
    """The headline AUC must be a held-out number, and lower than in-sample."""
    X, y, names, _ = make_synthetic(seed=3)
    out = run_rca(X, y, names)
    assert len(out["auc_folds"]) == 5
    assert out["auc"] <= out["auc_in_sample"] + 1e-9
    assert abs(out["auc"] - float(np.mean(out["auc_folds"]))) < 1e-9


def test_folds_are_disjoint_and_stratified():
    y = np.array([0] * 90 + [1] * 10)
    folds = stratified_folds(y, n_splits=5, seed=0)
    seen = np.concatenate([te for _, te in folds])
    assert len(seen) == len(y) and len(set(seen.tolist())) == len(y)
    for tr, te in folds:
        assert not set(tr.tolist()) & set(te.tolist())
        assert y[te].sum() == 2


def test_roc_auc_matches_known_values():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert roc_auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_cv_auc_beats_chance_on_signal():
    X, y, _, _ = make_synthetic(seed=4)
    auc, folds = cv_auc(LogisticRCA, X, y, n_splits=5, seed=0)
    assert auc > 0.7 and len(folds) == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("ok")
