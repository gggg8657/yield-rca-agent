"""Check that every number in WEEKEND.md still traces to a run in `runs/`.

`RESULTS.md` and the README are generated, so `scripts/report.py --check`
already guarantees they cannot go stale. `WEEKEND.md` is hand-written on
purpose -- it is the Monday-morning summary and it needs prose a generator
cannot produce -- which makes it the one document in this repository where a
number can drift away from the run that produced it. This script closes that
gap.

Each entry below names a claim, the JSON it must come from, and the exact string
that must appear in `WEEKEND.md`. The check fails if the JSON no longer produces
that string, or if the string is no longer in the document -- so both a stale
number and a silently deleted one are caught.

Typographic minus signs are normalised before matching: the document uses
U+2212 in tables and the formatters emit ASCII, and a mismatch there is a
rendering detail rather than a factual error.

    python scripts/audit_weekend.py          # exits 1 on any drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(runs: Path, name: str):
    p = runs / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def claims(d):
    """(label, source json, expected string) for every number in WEEKEND.md.

    Adding a number to WEEKEND.md means adding a row here. That is deliberate
    friction: it is the step that makes the number traceable.
    """
    ev, st = d.get("secom_eval"), d.get("secom_stability")
    nf, nf5 = d.get("null_fdr"), d.get("null_fdr_k5")
    ab, ab5 = d.get("abstain"), d.get("abstain_k5")
    rk, iv = d.get("null_fdr_rankers"), d.get("invariance")
    out = []

    if ev:
        a = ev["auc"]["per_arm"]
        p = ev["auc"]["paired"]["agent_rf__vs__rf_all"]
        out += [
            ("rf_all CV AUC", "secom_eval",
             f"{a['rf_all']['mean']:.3f} [{a['rf_all']['ci_lo']:.3f}, "
             f"{a['rf_all']['ci_hi']:.3f}]"),
            ("agent_rf CV AUC", "secom_eval",
             f"{a['agent_rf']['mean']:.3f} [{a['agent_rf']['ci_lo']:.3f}, "
             f"{a['agent_rf']['ci_hi']:.3f}]"),
            ("agent - baseline paired", "secom_eval",
             f"{p['mean']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]"),
        ]
    if st:
        rk_ = st["rankers"]
        def _stab(name):
            return 100 * rk_[name]["bootstrap"]["raw"]["pairwise_overlap"]
        def _wall(name):
            return rk_[name]["bootstrap"]["wall_min"]
        out.append(("agent top-5 stability", "secom_stability",
                    f"{_stab('agent'):.1f}%"))
        # The result-6 decomposition: every cell of the 2x2 and both deltas,
        # so the "+13.0 points for one field, about a point for the
        # architecture" claim cannot drift away from the run behind it.
        for name in ("univariate", "rf_impurity", "agent_model", "perm_only"):
            if name in rk_:
                out.append((f"{name} top-5 stability", "secom_stability",
                            f"{_stab(name):.1f}%"))
        if "agent_model" in rk_:
            out += [
                ("attribution delta (agent -> agent_model)", "secom_stability",
                 f"+{_stab('agent_model') - _stab('agent'):.1f}"),
                ("machinery delta on model attribution", "secom_stability",
                 f"-{_stab('rf_impurity') - _stab('agent_model'):.1f}"),
                ("machinery delta on permutation attribution",
                 "secom_stability",
                 f"+{_stab('agent') - _stab('perm_only'):.1f}"),
                ("agent_model speedup", "secom_stability",
                 f"{_wall('agent') / _wall('agent_model'):.1f}x"),
                ("agent_model wall", "secom_stability",
                 f"{_wall('agent_model'):.1f} min"),
                # The claim is a *ratio* of walls, and writing it from the
                # rounded minutes is how "14.6x" got into this document once.
                ("agent_model vs rf_impurity wall ratio", "secom_stability",
                 f"{_wall('agent_model') / _wall('rf_impurity'):.1f}x"),
                ("agent_model shortfall against rf_impurity",
                 "secom_stability",
                 f"{_stab('rf_impurity') - _stab('agent_model'):.1f} points"),
                ("agent wall", "secom_stability", f"{_wall('agent'):.1f} min"),
            ]
    if nf:
        out += [
            ("false discoveries on the null", "null_fdr",
             f"{nf['null_fdr']['false_discoveries_total']:,}"),
            ("noise sensors clearing pi on merit, k=40", "null_fdr",
             f"{nf['null']['n_merit_mean']:.1f}"),
        ]
    if nf5:
        out += [
            ("noise sensors clearing pi on merit, k=5", "null_fdr_k5",
             f"{nf5['null']['n_merit_mean']:.2f}"),
            ("fallback fired, k=5", "null_fdr_k5",
             f"{100*nf5['null']['fallback_rate']:.1f}%"),
        ]
    if nf5:
        # The 6b retraction rests on a cross-tab, not on a summary field, so
        # it is recomputed from the per-replicate records here exactly as
        # `report.fallback_reach` does it.
        null5 = [r for r in nf5["records"] if r["permuted"]]
        tau5 = nf5["thresholds"]["alpha_0.05"]
        fired = [r for r in null5 if r["fallback_fired"]]
        unaided = [r for r in null5 if not r["fallback_fired"]]
        if fired and unaided:
            over = lambda g: sum(
                any(v >= tau5 for v in r["stability_values"]) for r in g)
            out += [
                ("guard cross-tab: replicates it fired on", "null_fdr_k5",
                 f"| {len(fired)} | "
                 f"{max(r['max_stability'] for r in fired):.3f} |"),
                ("guard cross-tab: those naming a suspect over tau",
                 "null_fdr_k5", f"Zero of the {len(fired)} name anything"),
                ("guard cross-tab: replicates clearing pi on merit",
                 "null_fdr_k5",
                 f"| {len(unaided)} | "
                 f"{max(r['max_stability'] for r in unaided):.3f} | "
                 f"{over(unaided)} |"),
                ("tau(0.05) at k=5", "null_fdr_k5", f"{tau5:.3f}"),
            ]
    if ab:
        m = ab["levels"]["alpha_0.05"]
        out += [
            ("k=40 error control", "abstain",
             f"{100*m['null_abstention_heldout']:.1f}%"),
            ("k=40 suspects reported", "abstain", f"{m['real_reported_mean']:.2f}"),
        ]
        ns = ab.get("null_structure")
        if ns:
            out += [
                ("null replicate agreement", "abstain",
                 f"{ns['null']['top5_pairwise_overlap']:.3f}"),
                ("random-ranker floor", "abstain",
                 f"{ns['random_floor_top5']:.3f}"),
            ]
    if ab5:
        m = ab5["levels"]["alpha_0.05"]
        out += [
            ("k=5 error control", "abstain_k5",
             f"{100*m['null_abstention_heldout']:.1f}%"),
            ("k=5 suspects reported", "abstain_k5", f"{m['real_reported_mean']:.2f}"),
        ]
    if rk:
        per = rk["per_ranker"]
        # the same arm the report names: the best-controlled non-saturating
        # variant, not merely the first one in dict order
        cands = [v for v in per.values()
                 if v.get("is_variant") and v.get("select_k") == 5
                 and v.get("ranker") == "univariate"]
        uni = max(cands,
                  key=lambda v: v["heldout_alpha_0.05"]["null_abstention_heldout"]) \
            if cands else None
        ag = next((v for k, v in per.items() if k.startswith("agent")), None)
        if uni:
            out += [
                ("univariate k=5 error control", "null_fdr_rankers",
                 f"{100*uni['heldout_alpha_0.05']['null_abstention_heldout']:.1f}%"),
                ("univariate k=5 suspects", "null_fdr_rankers",
                 f"{uni['heldout_alpha_0.05']['real_reported_mean']:.2f}"),
            ]
        if ag:
            out.append(("agent separation", "null_fdr_rankers",
                        f"{ag['prob_real_max_exceeds_null_max']:.3f}"))
        if "univariate" in per:
            out.append(("univariate matched separation", "null_fdr_rankers",
                        f"{per['univariate']['prob_real_max_exceeds_null_max']:.3f}"))
    if iv:
        t = iv["totals"]
        out += [
            ("sensors associated with failure", "invariance",
             f"{t['n_associated']} of {t['n_sensors']}"),
            ("non-invariant sensors", "invariance",
             f"**{t['n_non_invariant']}**"),
        ]
    return out


def norm(s: str) -> str:
    return s.replace("−", "-").replace("–", "-").replace("—", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--doc", default=str(ROOT / "WEEKEND.md"))
    a = ap.parse_args()

    doc_p = Path(a.doc)
    if not doc_p.exists():
        print(f"{a.doc} does not exist yet; nothing to audit")
        return 0
    doc = norm(doc_p.read_text())

    runs = Path(a.runs)
    names = ["secom_eval", "secom_stability", "null_fdr", "null_fdr_k5",
             "abstain", "abstain_k5", "null_fdr_rankers", "invariance"]
    d = {n: load(runs, n) for n in names}
    missing = [n for n in names if d[n] is None]

    rows = claims(d)
    bad = [(lbl, src, exp) for lbl, src, exp in rows if norm(exp) not in doc]
    print(f"audited {len(rows)} numeric claims in {doc_p.name} "
          f"against {len(names) - len(missing)} run JSONs")
    if missing:
        print(f"  (skipped, JSON absent: {', '.join(missing)})")
    for lbl, src, exp in bad:
        print(f"  DRIFT  {lbl}: runs/{src}.json produces \"{exp}\", "
              f"which is not in {doc_p.name}")
    if bad:
        print(f"\n{len(bad)} claim(s) in {doc_p.name} no longer match the runs. "
              f"Either the document is stale or a run was replaced.")
        return 1
    print("  every audited number traces to a run in runs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
