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
import re
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
    nf5m, ab5m = d.get("null_fdr_k5_model"), d.get("abstain_k5_model")
    nfm, abm = d.get("null_fdr_model"), d.get("abstain_model")
    out = []

    if ev and ev.get("chronological", {}).get("rf_all"):
        out.append(("rf_all chronological AUC", "secom_eval",
                    f"{ev['chronological']['rf_all']['auc']:.3f}"))
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
                # The only unconfounded cell of the 2x2: one config field.
                ("attribution delta (agent -> agent_model)", "secom_stability",
                 f"+{_stab('agent_model') - _stab('agent'):.1f}"),
                ("machinery delta on permutation attribution",
                 "secom_stability",
                 f"+{_stab('agent') - _stab('perm_only'):.1f}"),
                ("agent_model speedup", "secom_stability",
                 f"{_wall('agent') / _wall('agent_model'):.1f}x"),
                ("agent_model wall", "secom_stability",
                 f"{_wall('agent_model'):.1f} min"),
                ("agent wall", "secom_stability", f"{_wall('agent'):.1f} min"),
            ]
        # The model-native architecture delta is only auditable once the
        # matched bare-ranker arm exists. Before that the document must carry
        # a blank, and this asserts the blank is there -- so a confounded
        # estimate cannot quietly reappear in its place.
        if "agent_model" in rk_ and "model_only" not in rk_:
            out.append(("model-native architecture delta is unmeasured",
                        "secom_stability", "*[not measured]*"))
        elif "model_only" in rk_:
            out += [
                ("model_only top-5 stability", "secom_stability",
                 f"{_stab('model_only'):.1f}%"),
                ("machinery delta on model attribution, matched arm",
                 "secom_stability",
                 f"{_stab('agent_model') - _stab('model_only'):+.1f}"),
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
    if ab5:
        # The strengthened version of the same claim: counted per
        # calibration/evaluation split rather than against one full-null tau.
        for key in ("alpha_0.1", "alpha_0.05", "alpha_0.01"):
            m = (ab5.get("levels") or {}).get(key)
            if not m or "tau_min" not in m:
                continue
            out += [
                (f"smallest tau fitted, {key}", "abstain_k5",
                 f"{m['tau_min']:.3f}"),
                (f"guard reached the report, {key}", "abstain_k5",
                 f"{m['fallback_reached_report_total']} of "
                 f"{m['n_splits_evaluated']}"),
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
    # H6: the same one-field change on the error-control axis. Both arms of
    # every comparison come from the same depth, because mixing depths is how
    # "separation 0.873 -> 0.994" nearly got written down.
    for tag, nfx, abx in (("k5", nf5m, ab5m), ("k40", nfm, abm)):
        if not (nfx and abx):
            continue
        for key in ("alpha_0.1", "alpha_0.05", "alpha_0.01"):
            m = (abx.get("levels") or {}).get(key)
            if not m:
                continue
            out += [
                (f"model attribution control, {tag} {key}",
                 f"abstain{'_k5' if tag == 'k5' else ''}_model",
                 f"{100 * m['null_abstention_heldout']:.1f}%"),
                (f"model attribution suspects, {tag} {key}",
                 f"abstain{'_k5' if tag == 'k5' else ''}_model",
                 f"{m['real_reported_mean']:.2f}"),
            ]
        out.append((f"model attribution separation, {tag}",
                    f"null_fdr{'_k5' if tag == 'k5' else ''}_model",
                    f"{nfx['separation']['prob_real_max_exceeds_null_max']:.3f}"))
    # And the permutation side of the same-depth pair, so a mixed-protocol
    # subtraction would break the audit rather than read plausibly.
    # Both alpha ladders of the permutation arm too. These were the two
    # numbers I typed from estimate into WEEKEND.md's k=40 table (0.79 and
    # 0.40, against the runs' 1.18 and 0.22) and the audit had no claim
    # covering them, so it passed. Now it does not.
    if nf and ab:
        for key in ("alpha_0.1", "alpha_0.05", "alpha_0.01"):
            m = (ab.get("levels") or {}).get(key)
            if m:
                out += [
                    (f"permutation control, k40 {key}", "abstain",
                     f"{100 * m['null_abstention_heldout']:.1f}%"),
                    (f"permutation suspects, k40 {key}", "abstain",
                     f"{m['real_reported_mean']:.2f}"),
                ]
    if nf5 and ab5:
        for key in ("alpha_0.1", "alpha_0.05", "alpha_0.01"):
            m = (ab5.get("levels") or {}).get(key)
            if m:
                out += [
                    (f"permutation control, k5 {key}", "abstain_k5",
                     f"{100 * m['null_abstention_heldout']:.1f}%"),
                    (f"permutation suspects, k5 {key}", "abstain_k5",
                     f"{m['real_reported_mean']:.2f}"),
                ]
        out.append(("permutation separation, k5", "null_fdr_k5",
                    f"{nf5['separation']['prob_real_max_exceeds_null_max']:.3f}"))
    # H7 (runs/attr_arm.json) and the PredictAllReportFew correction
    # (runs/par_few.json), both added under the published CV protocol.
    aa = d.get("attr_arm")
    if aa:
        out.append(("agent_model_rf AUC", "attr_arm",
                    f"{aa['auc']['mean']:.3f} [{aa['auc']['ci_lo']:.3f}, "
                    f"{aa['auc']['ci_hi']:.3f}]"))
        for arm in ("rf_all", "univar_top25_rf", "agent_rf"):
            pd_ = (aa.get("paired") or {}).get(f"{aa['arm']}__vs__{arm}")
            if pd_:
                out += [
                    (f"H7 paired vs {arm}", "attr_arm",
                     f"{pd_['mean']:+.4f} [{pd_['ci_lo']:+.4f}, "
                     f"{pd_['ci_hi']:+.4f}]"),
                    (f"H7 wilcoxon vs {arm}", "attr_arm",
                     f"{pd_['wilcoxon_p']:.3f}"),
                ]
        out.append(("H7 n_selected", "attr_arm",
                    f"{aa['n_selected_mean']:.1f}"))
    # The n_boot ladder read out of the ranker run: attainable-set sizes and
    # the closest-to-nominal value, both recomputed from its own records.
    if rk and rk.get("records"):
        import numpy as _np
        fam = sorted(((k, v) for k, v in (rk.get("per_ranker") or {}).items()
                      if v.get("is_variant") and v.get("select_k") == 5
                      and v.get("ranker") == "univariate"),
                     key=lambda kv: kv[1]["n_boot"])
        for name, v in fam:
            nb = v["n_boot"]
            mx = _np.asarray([r["max_stability"] for r in rk["records"]
                              if r["arm"] == name and r["permuted"]],
                             dtype=float)
            if not len(mx):
                continue
            ach = sorted({float(1.0 - (mx >= k / nb).mean())
                          for k in range(1, nb + 1)}, reverse=True)
            near = min(ach, key=lambda a: abs(a - 0.95))
            out += [
                (f"n_boot={nb} attainable count", "null_fdr_rankers",
                 f"| {nb} | 0.0% | {len([a for a in ach if a > 0.60])} |"),
                (f"n_boot={nb} closest to nominal", "null_fdr_rankers",
                 f"{near:.3f}"),
                (f"n_boot={nb} control", "null_fdr_rankers",
                 f"{100 * v['heldout_alpha_0.05']['null_abstention_heldout']:.1f}%"),
            ]
    # H9 and the grid/calibration decomposition.
    nfb, abb = d.get("null_fdr_k5_model_b40"), d.get("abstain_k5_model_b40")
    if nfb and abb:
        m = abb["levels"]["alpha_0.05"]
        out += [
            ("H9 control", "abstain_k5_model_b40",
             f"{100 * m['null_abstention_heldout']:.1f}%"),
            ("H9 suspects", "abstain_k5_model_b40",
             f"{m['real_reported_mean']:.2f}"),
            ("H9 separation", "null_fdr_k5_model_b40",
             f"{nfb['separation']['prob_real_max_exceeds_null_max']:.3f}"),
        ]
    cs = d.get("calib_size")
    if cs:
        for lab, v in (cs.get("arms") or {}).items():
            m100 = v["curve"].get("100") or {}
            out += [
                (f"oracle: {lab}", "calib_size",
                 f"{v['oracle_control']:.3f}"),
                (f"calibration loss: {lab}", "calib_size",
                 f"{m100.get('calibration_loss', float('nan')):+.3f}"),
            ]
    # Deltas quoted in the summary table. Derived, so computed from the raw
    # values rather than from the rounded percentages they are printed as --
    # that mistake put "-1.2" and "14.6x" into this document once.
    if ab5m and ab5 and (ab5m.get("levels") or {}).get("alpha_0.05"):
        d5 = (100 * ab5m["levels"]["alpha_0.05"]["null_abstention_heldout"]
              - 100 * ab5["levels"]["alpha_0.05"]["null_abstention_heldout"])
        out.append(("attribution delta on control, k=5", "abstain_k5_model",
                    f"{d5:+.1f}"))
    if abm and ab and (abm.get("levels") or {}).get("alpha_0.05"):
        d40 = (100 * abm["levels"]["alpha_0.05"]["null_abstention_heldout"]
               - 100 * ab["levels"]["alpha_0.05"]["null_abstention_heldout"])
        out.append(("attribution delta on control, k=40", "abstain_model",
                    f"{d40:+.1f}"))
    # --- systematic families -------------------------------------------
    # The paper quotes numbers RESULTS.md generates, by hand. Rather than
    # registering them one at a time as they appear, register the families
    # wholesale: every alpha level of every abstain file, every attainable
    # grid value of every null_fdr file, and the profile constants.
    import numpy as _np2
    for nm in ("abstain", "abstain_k5", "abstain_model", "abstain_k5_model",
               "abstain_k5_model_b40"):
        av = d.get(nm)
        if not av:
            continue
        for key, lv in (av.get("levels") or {}).items():
            out += [
                (f"{nm}/{key} control", nm,
                 f"{100 * lv['null_abstention_heldout']:.1f}%"),
                (f"{nm}/{key} suspects", nm, f"{lv['real_reported_mean']:.2f}"),
                (f"{nm}/{key} real abstention", nm,
                 f"{100 * lv['real_abstention']:.1f}%"),
                (f"{nm}/{key} tau", nm, f"{lv['tau_mean']:.3f}"),
            ]
        base = av.get("no_rule_baseline") or {}
        if "real_reported_mean" in base:
            out.append((f"{nm} no-rule suspects", nm,
                        f"{base['real_reported_mean']:.1f}"))
    for nm in ("null_fdr", "null_fdr_k5", "null_fdr_model",
               "null_fdr_k5_model", "null_fdr_k5_model_b40"):
        nv = d.get(nm)
        if not nv or not nv.get("records"):
            continue
        nb = ((nv.get("protocol") or {}).get("agent_cfg") or {}).get("n_boot")
        mx = _np2.asarray([r["max_stability"] for r in nv["records"]
                           if r["permuted"]], dtype=float)
        if nb and len(mx):
            for k in range(1, nb + 1):
                out.append((f"{nm} attainable k={k}", nm,
                            f"{1.0 - (mx >= k / nb).mean():.3f}"))
            out.append((f"{nm} P(M=1)", nm,
                        f"{100 * (mx >= 1.0).mean():.1f}%"))
        sep = (nv.get("separation") or {}).get("prob_real_max_exceeds_null_max")
        if sep is not None:
            out.append((f"{nm} separation", nm, f"{sep:.3f}"))
    rkj = d.get("null_fdr_rankers")
    if rkj and rkj.get("records"):
        for name, v in (rkj.get("per_ranker") or {}).items():
            nb = v.get("n_boot")
            mx = _np2.asarray([r["max_stability"] for r in rkj["records"]
                               if r["arm"] == name and r["permuted"]],
                              dtype=float)
            if not (nb and len(mx)):
                continue
            for k in range(1, nb + 1):
                out.append((f"{name} attainable k={k}", "null_fdr_rankers",
                            f"{1.0 - (mx >= k / nb).mean():.3f}"))
            out += [
                (f"{name} P(M=1)", "null_fdr_rankers",
                 f"{100 * (mx >= 1.0).mean():.1f}%"),
                (f"{name} separation", "null_fdr_rankers",
                 f"{v['prob_real_max_exceeds_null_max']:.3f}"),
            ]
    if st:
        rr = st["rankers"]
        for n1 in rr:
            for n2 in rr:
                if n1 >= n2:
                    continue
                dlt = (100 * rr[n1]["bootstrap"]["raw"]["pairwise_overlap"]
                       - 100 * rr[n2]["bootstrap"]["raw"]["pairwise_overlap"])
                out.append((f"stability delta {n1}-{n2}", "secom_stability",
                            f"{abs(dlt):.1f}"))
    prof = d.get("data_profile")
    if prof:
        for k, v in prof.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((f"profile {k}", "data_profile",
                            f"{v:.3f}" if isinstance(v, float) else str(v)))
    dr = d.get("drift")
    if dr:
        def _walk(o, pre=""):
            if isinstance(o, dict):
                for k2, v2 in o.items():
                    yield from _walk(v2, f"{pre}.{k2}")
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                yield pre, o
        for k2, v2 in _walk(dr):
            if isinstance(v2, float):
                out.append((f"drift {k2}", "drift", f"{v2:.3f}"))
    # tau bounds across splits, and every scalar the invariance run reports.
    for nm in ("abstain", "abstain_k5", "abstain_model", "abstain_k5_model",
               "abstain_k5_model_b40"):
        av = d.get(nm)
        for key, lv in ((av or {}).get("levels") or {}).items():
            for f in ("tau_min", "tau_max"):
                if f in lv:
                    out.append((f"{nm}/{key} {f}", nm, f"{lv[f]:.3f}"))
    if iv:
        def _scalars(o, pre=""):
            if isinstance(o, dict):
                for k2, v2 in o.items():
                    yield from _scalars(v2, f"{pre}.{k2}")
            elif isinstance(o, list):
                for v2 in o:
                    yield from _scalars(v2, pre)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                yield pre, o
        seen = set()
        for k2, v2 in _scalars(iv):
            for fmt in ("{:.3f}", "{:.2f}", "{:.1f}", "{:.0f}%"):
                try:
                    t = fmt.format(v2 * 100 if fmt.endswith("%%") else v2)
                except (TypeError, ValueError):
                    continue
                if t not in seen:
                    seen.add(t)
                    out.append((f"invariance {k2}", "invariance", t))
            if isinstance(v2, float) and 0.0 <= v2 <= 1.0:
                t = f"{100 * v2:.1f}%"
                if t not in seen:
                    seen.add(t)
                    out.append((f"invariance {k2} pct", "invariance", t))
    if st:
        for n1, v1 in st["rankers"].items():
            b1 = v1["bootstrap"]["raw"]
            out += [
                (f"{n1} sd", "secom_stability",
                 f"{100 * b1['pairwise_overlap_sd']:.1f}"),
                (f"{n1} KPI gap", "secom_stability",
                 f"{80 - 100 * b1['pairwise_overlap']:.1f}"),
            ]
            w1 = v1["bootstrap"]["wall_min"]
            for n2, v2 in st["rankers"].items():
                w2 = v2["bootstrap"]["wall_min"]
                if w2 > 0:
                    out.append((f"wall ratio {n1}/{n2}", "secom_stability",
                                f"{w1 / w2:.1f}x"))
    # Named derived quantities the paper quotes. Registered individually and
    # with their provenance in the label, rather than by widening the wholesale
    # families above -- the haystack is already large enough that a coincidental
    # match is a real possibility, and every additional bulk family makes
    # "traces to a run" weaker evidence.
    for nm in ("null_fdr", "null_fdr_k5", "null_fdr_model",
               "null_fdr_k5_model", "null_fdr_k5_model_b40"):
        nv = d.get(nm)
        if not nv:
            continue
        w = (nv.get("environment") or {}).get("wall_min")
        if w:
            out.append((f"{nm} wall", nm, f"{w:.1f}"))
        for side in ("null", "real"):
            sv = nv.get(side) or {}
            for f, fmt in (("n_merit_mean", "{:.1f}"),
                           ("max_stability_mean", "{:.3f}"),
                           ("n_reported_mean", "{:.2f}")):
                if f in sv:
                    out.append((f"{nm}/{side} {f}", nm, fmt.format(sv[f])))
    for nm in ("abstain", "abstain_k5", "abstain_model", "abstain_k5_model",
               "abstain_k5_model_b40"):
        av = d.get(nm)
        base = (av or {}).get("no_rule_baseline") or {}
        if "real_reported_mean" in base:
            out.append((f"{nm} no-rule suspects (2dp)", nm,
                        f"{base['real_reported_mean']:.2f}"))
    if iv and iv.get("per_sensor"):
        import numpy as _np3
        strengths = [abs(x["pooled_auc"] - 0.5) for x in iv["per_sensor"]
                     if x.get("associated") and "pooled_auc" in x]
        if strengths:
            out += [
                ("association strength min", "invariance",
                 f"{min(strengths):.3f}"),
                ("association strength max", "invariance",
                 f"{max(strengths):.3f}"),
                ("association strength median", "invariance",
                 f"{float(_np3.median(strengths)):.3f}"),
            ]
    # Separation gaps against the univariate arm, and the superseded stability
    # delta the paper quotes when recording its own retraction.
    rkj2 = d.get("null_fdr_rankers")
    if rkj2:
        uni2 = next((v for k, v in (rkj2.get("per_ranker") or {}).items()
                     if k.startswith("univariate (n_boot=40, select_k=5")), None)
        if uni2:
            us = uni2["prob_real_max_exceeds_null_max"]
            for nm in ("null_fdr_k5", "null_fdr_k5_model",
                       "null_fdr_k5_model_b40", "null_fdr", "null_fdr_model"):
                nv = d.get(nm)
                if not nv:
                    continue
                sep = (nv.get("separation") or {}).get(
                    "prob_real_max_exceeds_null_max")
                if sep is not None:
                    out.append((f"separation gap univariate-{nm}",
                                "null_fdr_rankers", f"{us - sep:.3f}"))
    if st and "rf_impurity" in st["rankers"] and "agent_model" in st["rankers"]:
        rr2 = st["rankers"]
        dlt = (100 * rr2["agent_model"]["bootstrap"]["raw"]["pairwise_overlap"]
               - 100 * rr2["rf_impurity"]["bootstrap"]["raw"]["pairwise_overlap"])
        out.append(("superseded architecture delta (rf_impurity cell)",
                    "secom_stability", f"{dlt:+.1f}"))
    if ab and ab5 and (ab.get("levels") or {}).get("alpha_0.05"):
        dd5 = (100 * ab5["levels"]["alpha_0.05"]["null_abstention_heldout"]
               - 100 * ab["levels"]["alpha_0.05"]["null_abstention_heldout"])
        out += [("depth delta on control", "abstain_k5", f"{dd5:+.1f}"),
                ("depth delta on control (2dp)", "abstain_k5", f"{dd5:+.2f}")]
    dd = d.get("dedup")
    if dd:
        for th, v in (dd.get("verdicts") or {}).items():
            out += [
                (f"dedup loop families @{th}", "dedup",
                 f"{v['loop_families']:.3f}"),
                (f"dedup univariate families @{th}", "dedup",
                 f"{v['univariate_families']:.3f}"),
            ]
        for th, v in (dd.get("by_threshold") or {}).items():
            out.append((f"dedup n_families @{th}", "dedup",
                        str(v["n_families"])))
    sp = d.get("sparsity")
    if sp:
        for k in sp["caps"]:
            pm = sp["curves"]["permutation"][str(k)]
            mm = sp["curves"]["model"][str(k)]
            dd = sp["per_rung"][str(k)]["model_minus_permutation"]
            out += [
                (f"sparsity perm AUC cap {k}", "sparsity",
                 f"{pm['auc']['mean']:.4f}"),
                (f"sparsity model AUC cap {k}", "sparsity",
                 f"{mm['auc']['mean']:.4f}"),
                (f"sparsity delta cap {k}", "sparsity", f"{dd['mean']:+.4f}"),
            ]
        sl = (sp.get("sparsity_slope") or {}).get("model")
        if sl:
            out.append(("sparsity slope", "sparsity",
                        f"{sl['auc_per_sensor']:+.5f}"))
    pf = d.get("par_few")
    if pf:
        u = pf["per_arm"]["par_untuned"]["auc"]
        dl = pf["paired_tuned_minus_untuned"]
        out += [
            ("par_untuned AUC", "par_few",
             f"{u['mean']:.3f} [{u['ci_lo']:.3f}, {u['ci_hi']:.3f}]"),
            ("par tuned - untuned", "par_few",
             f"{dl['mean']:+.4f} [{dl['ci_lo']:+.4f}, {dl['ci_hi']:+.4f}]"),
        ]
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
    return s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")


# Numbers that are not measurements: dates, structural counts, thresholds and
# parameter values that name a configuration rather than report a result.
# Everything here is a literal that a run does not produce and should not be
# expected to. Keep it short and justified -- it is the escape hatch, and a
# long escape hatch defeats the check.
ALLOW = {
    # dates and times in prose
    "2026", "09", "05", "04", "03", "22", "08", "30", "40", "12", "5", "15",
    # section numbering and list counts
    "1", "2", "3", "4", "6", "7", "8", "9", "10", "11", "13", "14",
    # configuration values that name an arm rather than report a measurement
    "0.9", "0.90", "0.95", "0.99", "0.05", "0.01", "0.1", "0.10",
    "25", "20", "100",
    "50", "200", "474", "104", "1567", "590", "80", "0.759", "0.75",
    # protocol constants and a citation year: chosen inputs, not measurements
    "1463", "2016", "20000", "400", "60", "0.5", "0.7", "0.3",
}

# A number is a measurement candidate only when it stands on its own. Excluded
# by construction: ISO timestamps, `sensor_059`-style identifiers, and the
# trailing digit of hyphenated words like "top-5" or "half-and-half".
ISO = re.compile(r"\d{4}-\d{2}-\d{2}(?:T[\d:+]+)?")
IDENT = re.compile(r"[A-Za-z_]\w*_\d+")
NUM = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?%?")
THOUS = re.compile(r"\d{1,3}(?:,\d{3})+")
HEAD = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)?)[.\s]")


def coverage(doc: str, rows) -> list:
    """Numeric literals in the document that no registered claim accounts for.

    The registration check below catches a claim whose value has drifted. It
    cannot catch a number nobody registered -- and this weekend two such
    numbers were typed from estimate into a table and the audit passed. So the
    complementary direction is checked too: every number in the document must
    appear inside some string a run currently produces.

    Code blocks are skipped: they carry commands and flags, not results.
    """
    body, in_code = [], False
    for line in doc.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            body.append(line)
    text = "\n".join(body)
    text = ISO.sub(" ", text)
    text = IDENT.sub(" ", text)
    # "1,567" must not read as 567. Normalise thousands separators on both
    # sides rather than only one, or the comparison silently misses.
    text = THOUS.sub(lambda m: m.group(0).replace(",", ""), text)

    # A number that is one of this document's own section headings is
    # structural. Collected from the document rather than hard-coded, so a
    # renumbered draft does not start failing.
    heads = set()
    for line in body:
        m = HEAD.match(line)
        if m:
            heads.add(m.group(1))
            heads.update(m.group(1).split("."))

    haystack = " || ".join(norm(exp) for _, _, exp in rows)
    haystack = THOUS.sub(lambda m: m.group(0).replace(",", ""), haystack)
    missing = {}
    for tok in NUM.findall(norm(text)):
        bare = tok.rstrip("%")
        if tok in ALLOW or bare in ALLOW or bare in heads:
            continue
        if tok in haystack or bare in haystack:
            continue
        missing.setdefault(tok, 0)
        missing[tok] += 1
    return sorted(missing.items(), key=lambda kv: -kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--doc", action="append", default=None,
                    help="hand-written document to audit; repeatable. "
                         "Defaults to WEEKEND.md and paper_draft.md -- "
                         "RESULTS.md and the README are generated and are "
                         "guarded by scripts/report.py --check instead.")
    ap.add_argument("--no-coverage", action="store_true",
                    help="skip the unregistered-number scan")
    a = ap.parse_args()

    docs = [Path(x) for x in (a.doc or [str(ROOT / "WEEKEND.md"),
                                         str(ROOT / "paper_draft.md")])]
    docs = [x for x in docs if x.exists()]
    if not docs:
        print("no hand-written documents found; nothing to audit")
        return 0

    runs = Path(a.runs)
    names = ["secom_eval", "secom_stability", "null_fdr", "null_fdr_k5",
             "abstain", "abstain_k5", "null_fdr_rankers", "invariance",
             "null_fdr_k5_model", "abstain_k5_model",
             "null_fdr_model", "abstain_model",
             "attr_arm", "par_few", "sparsity",
             "null_fdr_k5_model_b40", "abstain_k5_model_b40", "calib_size",
             "dedup", "data_profile", "drift"]
    d = {n: load(runs, n) for n in names}
    missing_json = [n for n in names if d[n] is None]
    rows = claims(d)
    if missing_json:
        print(f"(skipped, JSON absent: {', '.join(missing_json)})")

    rc = 0
    for doc_p in docs:
        doc = norm(doc_p.read_text())
        present = [r for r in rows if norm(r[2]) in doc]
        print(f"\n{doc_p.name}: {len(rows)} registered claims checked, "
              f"{len(present)} appear")
        if a.no_coverage:
            continue
        bad = coverage(doc, rows)
        if bad:
            print(f"  {len(bad)} numeric literal(s) match no value any run "
                  f"currently produces:")
            for tok, n in bad[:40]:
                print(f"    UNTRACED  {tok}   (x{n})")
            rc = 1
        else:
            print("  every numeric literal traces to a value a run produces")
    if rc:
        print("\nEither the number is stale, or it was typed rather than "
              "measured, or it is structural and belongs in ALLOW.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
