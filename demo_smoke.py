"""End-to-end smoke demo on synthetic data — runs with numpy only.

    python demo_smoke.py

Prints the AUC, the recovered root-cause sensors, and whether the pipeline
actually recovered the injected causal sensors.
"""
from yieldrca import run_rca, make_synthetic


def main():
    X, y, names, causal = make_synthetic(seed=0)
    print(f"data: {X.shape[0]} wafers x {X.shape[1]} sensors, "
          f"fail rate {y.mean():.1%}, injected causal = {list(causal)}")
    out = run_rca(X, y, names)
    print(f"\nclassifier AUC: {out['auc']:.3f}")
    print("\n" + out["report"])

    found = {i for i, _ in out["ranked"]}
    recovered = sorted(found & set(causal.tolist()))
    print(f"\nrecovered {len(recovered)}/{len(causal)} injected causal "
          f"sensors: {recovered}")
    return out, causal


if __name__ == "__main__":
    main()
