"""Smoke test: the pipeline runs and recovers most injected causal sensors."""
import numpy as np
from yieldrca import run_rca, make_synthetic


def test_runs_and_recovers():
    X, y, names, causal = make_synthetic(seed=1)
    out = run_rca(X, y, names)
    assert 0.0 <= out["auc"] <= 1.0
    assert out["auc"] > 0.75, f"AUC too low: {out['auc']}"
    assert len(out["ranked"]) >= 1
    found = {i for i, _ in out["ranked"]}
    hit = len(found & set(causal.tolist()))
    assert hit >= max(1, len(causal) // 2), f"recovered only {hit}/{len(causal)}"


def test_report_is_markdown():
    X, y, names, _ = make_synthetic(seed=2)
    out = run_rca(X, y, names)
    assert out["report"].startswith("# Yield Root-Cause Report")


if __name__ == "__main__":
    test_runs_and_recovers()
    test_report_is_markdown()
    print("ok")
