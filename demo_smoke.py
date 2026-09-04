"""End-to-end smoke demo on synthetic data — runs with numpy only.

    python demo_smoke.py

Prints a cross-validated AUC, the recovered root-cause sensors, and how many of
the injected causal sensors the pipeline found. The AUC printed is 5-fold
cross-validated; the in-sample number is printed beside it only to show how
misleading it is on 200 sensors and 1,500 wafers.

This is the *synthetic* benchmark: the causal set is known by construction, so
recovery is measurable. On real SECOM data no such ground truth exists — see
`scripts/eval_secom.py` and RESULTS.md.
"""
from yieldrca import make_synthetic, run_rca


def main():
    X, y, names, causal = make_synthetic(seed=0)
    print(f"data: {X.shape[0]} wafers x {X.shape[1]} sensors, "
          f"fail rate {y.mean():.1%}, injected causal = {list(causal)}")
    out = run_rca(X, y, names)
    print(f"\nclassifier AUC (5-fold CV): {out['auc']:.3f}"
          f"   [in-sample, for contrast: {out['auc_in_sample']:.3f}]")
    print("\n" + out["report"])

    found = {i for i, _ in out["ranked"]}
    recovered = sorted(found & set(causal.tolist()))
    print(f"\nrecovered {len(recovered)}/{len(causal)} injected causal "
          f"sensors: {recovered}")
    return out, causal


if __name__ == "__main__":
    main()
