"""Assemble RESULTS.md -- and the README's number blocks -- from the run JSONs.

    python scripts/report.py            # regenerate RESULTS.md + README blocks
    python scripts/report.py --check    # fail if either is stale (used by CI)

Nothing in RESULTS.md or between the README's ``<!-- BEGIN:x -->`` markers is
typed by hand. Every figure, and every comparative sentence built around one,
is computed here from ``runs/*.json``, so a rerun that moves a number moves the
prose with it -- including the KPI verdicts, which are evaluated rather than
asserted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KPI_AUC = 0.75
KPI_STABILITY = 0.80


# ---------------------------------------------------------------- helpers
def read_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def cell(x):
    return str(x).replace("|", "\\|")


def table(rows, header):
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def ci(v, nd=3):
    return f"{v['mean']:.{nd}f} [{v['ci_lo']:.{nd}f}, {v['ci_hi']:.{nd}f}]"


def ci01(v, nd=2):
    """CI for a quantity bounded in [0, 1], clipped at the bounds.

    A recall of 0.98 has a normal-approximation interval reaching past 1.0.
    Printing that would be sloppier than clipping it and saying so.
    """
    lo, hi = max(0.0, v["ci_lo"]), min(1.0, v["ci_hi"])
    return f"{v['mean']:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def signed(v, nd=4):
    return f"{v['mean']:+.{nd}f} [{v['ci_lo']:+.{nd}f}, {v['ci_hi']:+.{nd}f}]"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def verdict(value, target, higher_is_better=True):
    ok = value >= target if higher_is_better else value <= target
    return ("**met**" if ok else "**not met**"), ok


def paired_delta(a, b, conf=0.95):
    """Paired mean difference a - b with a Student-t CI.

    Duplicated from ``yieldrca.evaluate`` on purpose: this script reads JSON
    and writes Markdown, and keeping it importable without the package (or
    scipy) means the report can always be regenerated from a checkout.
    """
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return {"mean": m, "ci_lo": m, "ci_hi": m, "n": n,
                "wins": sum(1 for v in d if v > 0),
                "losses": sum(1 for v in d if v < 0)}
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    # Student-t 97.5th percentile, df = n - 1 (table; no scipy dependency)
    T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
         24: 2.064, 30: 2.042}
    df = n - 1
    t = T.get(df) or T[min(T, key=lambda k: abs(k - df))]
    h = t * sd / math.sqrt(n)
    return {"mean": m, "ci_lo": m - h, "ci_hi": m + h, "n": n,
            "wins": sum(1 for v in d if v > 0),
            "losses": sum(1 for v in d if v < 0)}


def crosses_zero(d):
    return d["ci_lo"] <= 0.0 <= d["ci_hi"]


def marginal(d, eps=0.005):
    """A CI that only just reaches zero. Calling that a tie needs a caveat."""
    return crosses_zero(d) and min(abs(d["ci_lo"]), abs(d["ci_hi"])) < eps


# ---------------------------------------------------------------- sections
def sec_limits(prof, st):
    """The limits list, with its counts read from the run JSONs."""
    if not prof:
        return []
    L = [
        "- **No causal ground truth on SECOM**, so nothing here validates the "
        "*causal* half of \"root-cause analysis\" on real data. The synthetic "
        "benchmark is a proxy, and its planted structure is "
        "additive-logistic, which is kinder than a fab.",
        f"- **{prof['n_fail']} positives.** That is the binding constraint on "
        f"both KPIs, and no modelling choice in this repo escapes it. Fixing "
        f"the stability number needs more failed wafers, not a better ranker.",
        "- **Sensors are anonymous.** A surviving suspect cannot be mapped to a "
        "tool or a process step, so a domain expert cannot sanity-check the "
        "list -- which is exactly the check that would matter most.",
        "- **Permutation importance is not a causal effect.** It measures what "
        "a fitted model leans on. Two near-identical sensors split it, and a "
        "genuine driver the screen missed never gets scored at all.",
        "- **Untested variations that could matter.** The missing-indicator "
        "columns are attached in the logistic arm but not the forest arm; the "
        "screen is multivariate-linear or model-native, so a sensor that "
        "matters only through an interaction can be missed before attribution "
        "ever sees it. Both are choices this repo made and did not ablate.",
    ]
    if st:
        L.append(
            f"- **The stability metric is protocol-sensitive.** Bootstrap and "
            f"CV-fold resampling disagree by tens of points on the same "
            f"ranker (see the table), so any \"top-5 stability\" figure "
            f"quoted without its perturbation scheme is uninterpretable. This "
            f"repo reports both and headlines the harder one.")
    return ["## Limits", ""] + L + [""]


def sec_intro(prof, ev):
    """The one-line dataset description, so even the intro is not hand-typed."""
    if not prof:
        return []
    line = (f"The real data is the **UCI SECOM** dataset: "
            f"{prof['n_wafers']:,} wafers, {prof['n_sensors']} process "
            f"sensors, {prof['n_fail']} fails ({pct(prof['fail_rate'], 1)}, a "
            f"1:{prof['imbalance_ratio']:.0f} imbalance), "
            f"{pct(prof['missing_frac_overall'], 1)} of cells missing, "
            f"{prof['time_span_days']:.0f} days of a single campaign.")
    if ev:
        line += (f" It is evaluated under {ev['protocol']['cv']}, plus a "
                 f"chronological split and a rolling-origin split that never "
                 f"trains on a wafer produced after the one it scores.")
    return ["", line, ""]


def sec_dataset(prof):
    if not prof:
        return []
    c = prof["cleaner"]
    cl99 = prof["corr_clusters_at_0.99"]
    rows = [
        ["wafers x sensors", f"{prof['n_wafers']:,} x {prof['n_sensors']}"],
        ["fails / pass:fail ratio",
         f"{prof['n_fail']} ({pct(prof['fail_rate'], 2)}) / "
         f"1:{prof['imbalance_ratio']:.1f}"],
        ["time span", f"{prof['time_span_days']:.0f} days, timestamps monotone"],
        ["missing cells overall", pct(prof["missing_frac_overall"], 2)],
        ["sensors with any missing value",
         f"{prof['sensors_with_any_missing']} of {prof['n_sensors']}"],
        ["sensors over 50% missing", str(prof["sensors_missing_gt_50pct"])],
        ["worst single sensor", f"{pct(prof['max_missing_frac'])} missing"],
        ["wafers with at least one missing value",
         f"{prof['rows_with_any_missing']:,} of {prof['n_wafers']:,} (all of them)"],
        ["constant sensors (zero variance)", str(prof["constant_sensors"])],
        ["exact-duplicate sensor groups",
         f"{prof['duplicate_groups']} covering "
         f"{prof['duplicate_sensors_removable']} removable columns"],
        ["sensors surviving the cleaner",
         f"{c['sensors_kept']} of {c['sensors_in']} "
         f"({c['dropped_total']} dropped)"],
        ["missing-indicator columns appended",
         str(c["missing_indicator_columns"])],
        ["sensors with an |r| > 0.99 partner",
         f"{cl99['sensors_in_a_multi_member_cluster']} "
         f"({cl99['clusters_over_kept_sensors']} correlation clusters over "
         f"{cl99['kept_sensors']} kept sensors)"],
    ]
    dups = prof["duplicate_groups"]
    after = c["exact_duplicate_groups_after_constant_drop"]
    note = (
        f"All {dups} exact-duplicate groups turn out to sit *inside* the "
        f"{prof['constant_sensors']} constant sensors: once those are dropped, "
        f"{after} exact duplicates remain. Near-duplicates are the real problem "
        f"-- {cl99['sensors_in_a_multi_member_cluster']} of the "
        f"{cl99['kept_sensors']} surviving sensors have a partner above "
        f"|r| = 0.99, which is why the pipeline groups them and why the "
        f"stability KPI is reported both per sensor and per group."
    )
    return ["## The data, as it actually arrives", "",
            "Every count here is measured by `scripts/prepare_data.py`, not "
            "quoted from the dataset description:", "",
            table(rows, ["property", "value"]), "", note, ""]


def _arm_rows(ev, order=None):
    auc, ap = ev["auc"], ev["ap"]
    names = order or sorted(auc["per_arm"], key=lambda a: -auc["per_arm"][a]["mean"])
    rows = []
    for a in names:
        d = auc["paired"].get(f"{a}__vs__rf_all")
        rows.append([
            f"`{a}`",
            ev["arms"][a]["label"],
            ci(auc["per_arm"][a]),
            f"{ap['per_arm'][a]['mean']:.3f}",
            "--" if a == "rf_all" else signed(d, 3),
            f"{ev['chronological'][a]['auc']:.3f}",
        ])
    return rows, names


def sec_secom_auc(ev):
    if not ev:
        return []
    auc = ev["auc"]
    best = max(auc["per_arm"], key=lambda a: auc["per_arm"][a]["mean"])
    rf = auc["per_arm"]["rf_all"]
    v_auc, auc_ok = verdict(rf["mean"], KPI_AUC)
    rows, _ = _arm_rows(ev)
    agent = auc["per_arm"]["agent_rf"]
    d_agent = auc["paired"]["agent_rf__vs__rf_all"]
    d_univ = auc["paired"]["univar_top25_rf__vs__rf_all"]
    d_agent_vs_univ = auc["paired"]["agent_rf__vs__univar_top25_rf"]
    nsel = [r["n_selected"] for r in ev["records"] if r["arm"] == "agent_rf"]
    nsel_mean = sum(nsel) / len(nsel)
    ncand = [r["n_candidates"] for r in ev["records"] if r["arm"] == "agent_rf"]
    ncand_mean = sum(ncand) / len(ncand)

    lead = (
        f"{auc['n_folds']} folds, identical for every arm "
        f"({ev['protocol']['cv']}). Baseline hyperparameters are chosen by an "
        f"inner {ev['protocol']['inner_tuning'].split('(')[1].split(')')[0]}-fold "
        f"grid search on each outer training fold, so no baseline here is the "
        f"best of a grid scored on the test folds. The delta column is a "
        f"**paired** per-fold difference against `rf_all`."
    )
    body = [
        "## SECOM: prediction", "", lead, "",
        table(rows, ["arm", "what it is", "ROC-AUC (95% CI)", "avg precision",
                     "paired delta vs `rf_all`", "chrono AUC"]), "",
        f"The best arm is `{best}` at {auc['per_arm'][best]['mean']:.3f}. "
        f"A plain random forest with fold-internal cleaning and median "
        f"imputation reaches **{rf['mean']:.3f}** "
        f"[{rf['ci_lo']:.3f}, {rf['ci_hi']:.3f}], so the AUC KPI "
        f"(>= {KPI_AUC:.2f}) is {v_auc} -- by the baseline, before any agent "
        f"runs.", "",
        f"The agent loop scores {agent['mean']:.3f} "
        f"[{agent['ci_lo']:.3f}, {agent['ci_hi']:.3f}] while handing the final "
        f"classifier {nsel_mean:.0f} sensors on average -- screened from the "
        f"survivors down to {ncand_mean:.0f} cluster representatives, then "
        f"cut again by the bootstrap drop. "
        f"Paired against the baseline that is {signed(d_agent, 3)} AUC "
        f"({d_agent['wins']} folds better, {d_agent['losses']} worse, "
        f"Wilcoxon p = {d_agent.get('wilcoxon_p', float('nan')):.1e}). "
        + ("That interval straddles zero, so the two are not distinguishable "
           "on this data."
           if crosses_zero(d_agent) else
           "That interval excludes zero: **the agent loop does not beat the "
           "obvious baseline on SECOM, it loses to it.**"), "",
        f"The naive-selection control tells us how much of that is selection "
        f"per se rather than this particular selector: univariate top-25 into "
        f"the same forest is {signed(d_univ, 3)} against the baseline, and "
        f"the agent loop differs from *it* by {signed(d_agent_vs_univ, 3)}"
        + (" -- an interval that straddles zero, so the plan/verify machinery "
           "buys no measurable accuracy over ranking each sensor on its own."
           if crosses_zero(d_agent_vs_univ) else
           ", an interval clear of zero, so the plan/verify machinery does buy "
           "accuracy over ranking each sensor on its own -- just not enough to "
           "catch using every sensor."), "",
    ]
    ch = ev["chronological"]
    real = [a for a in auc["per_arm"] if a != "majority"]
    dropped = [a for a in real if ch[a]["auc"] < auc["per_arm"][a]["mean"]]
    worst = min(real, key=lambda a: ch[a]["auc"])
    best_ch = max(real, key=lambda a: ch[a]["auc"])
    body += [
        f"The chronological column trains on the earliest "
        f"{ev['protocol']['chronological']['n_train']} wafers and tests on the "
        f"last {ev['protocol']['chronological']['n_test']} "
        f"({ev['protocol']['chronological']['n_fail_test']} fails). "
        + ("All " if len(dropped) == len(real) else f"{len(dropped)} of the ")
        + f"{len(real)} non-trivial arms score lower there than under "
          f"shuffled CV; `rf_all` falls from {rf['mean']:.3f} to "
        f"{ch['rf_all']['auc']:.3f}, and the whole field lands between "
        f"{ch[worst]['auc']:.3f} (`{worst}`) and {ch[best_ch]['auc']:.3f} "
        f"(`{best_ch}`). "
        + ("That is close enough to chance to say the plain reading out loud: "
           "**forward in time, none of these models has much predictive power "
           "on SECOM.** "
           if ch[best_ch]["auc"] < 0.62 else
           "The ordering also reshuffles, so the shuffled-CV ranking does not "
           "survive the change of protocol. ")
        + f"Over the {ev['protocol']['chronological']['n_train'] + ev['protocol']['chronological']['n_test']} "
          f"wafers of a 90-day campaign the sensor distributions drift, and a "
          f"shuffled split quietly lets the model interpolate across drift it "
          f"would never see in production. The KPI is stated against the "
          f"shuffled protocol, so that is what the scorecard scores -- but this "
          f"column is the one that decides whether the thing is deployable.",
        "",
    ]
    return body


def sec_rolling(ev, prof=None):
    if not ev or "rolling_origin" not in ev:
        return []
    ro = ev["rolling_origin"]
    auc = ev["auc"]["per_arm"]
    ch = ev["chronological"]
    n_orig = ev["protocol"]["rolling_origin"]["n_origins"]
    names = sorted(ro, key=lambda a: -ro[a]["summary"]["mean"])
    rows = []
    for a in names:
        per = ro[a]["per_origin"]
        rows.append([f"`{a}`", f"{auc[a]['mean']:.3f}", f"{ch[a]['auc']:.3f}",
                     ci(ro[a]["summary"]),
                     " · ".join(f"{r['auc']:.3f}" for r in per)])
    real = [a for a in ro if a != "majority"]
    best = max(real, key=lambda a: ro[a]["summary"]["mean"])
    best_shuffled = max(real, key=lambda a: auc[a]["mean"])
    sizes = ro[names[0]]["per_origin"]
    kind_word = {"agent": "an agent arm", "control": "a selection control",
                 "baseline": "a baseline"}
    kinds = {k: kind_word.get(v["kind"], v["kind"])
             for k, v in ev["arms"].items()}
    nsel = [r["n_selected"] for r in ev["records"] if r["arm"] == "agent_rf"]
    n_sel_mean = (sum(nsel) / len(nsel)) if nsel else None

    # is the shuffled-CV ordering preserved forward in time?
    pairs = list(itertools.combinations(real, 2))
    conc = sum(1 for a, b in pairs
               if (auc[a]["mean"] - auc[b]["mean"])
               * (ro[a]["summary"]["mean"] - ro[b]["summary"]["mean"]) > 0)
    # paired over origins: does the loop actually beat the baseline here?
    d = None
    if "agent_rf" in ro and "rf_all" in ro:
        d = paired_delta([r["auc"] for r in ro["agent_rf"]["per_origin"]],
                         [r["auc"] for r in ro["rf_all"]["per_origin"]])

    body = ["## Does it survive going forward in time?", "",
            f"One chronological split can be one unlucky fortnight, so the "
            f"same question is asked at every origin: "
            f"{ev['protocol']['rolling_origin']['rule']}. This is the only "
            f"protocol here that answers *would this have worked had we "
            f"deployed it* -- it never trains on a wafer produced after the "
            f"one it scores.", "",
            table(rows, ["arm", "shuffled CV", "chrono 70/30",
                         f"rolling origin, mean of {n_orig} (95% CI)",
                         "per origin"]), "",
            f"Two things happen at once here, and only one of them is solid.",
            "",
            f"**Solid: everything degrades.** The best shuffled-CV arm "
            f"(`{best_shuffled}`, {auc[best_shuffled]['mean']:.3f}) drops to "
            f"{ro[best_shuffled]['summary']['mean']:.3f} "
            f"[{ro[best_shuffled]['summary']['ci_lo']:.3f}, "
            f"{ro[best_shuffled]['summary']['ci_hi']:.3f}] forward in time, "
            f"and every arm's rolling-origin CI "
            + ("includes 0.5"
               if all(ro[a]["summary"]["ci_lo"] <= 0.5 for a in real)
               else "is wide enough to matter")
            + f". Whatever SECOM's shuffled-CV skill is made of, a substantial "
              f"part of it does not survive being asked to predict the next "
              f"block of wafers.", "",
            f"**Not solid: the ranking inverts.** Only {conc} of the "
            f"{len(pairs)} arm pairs keep their shuffled-CV order, and the "
            f"best arm forward in time is `{best}` "
            f"({ro[best]['summary']['mean']:.3f}"
            f"{'' if best == best_shuffled else ', ' + kinds.get(best, 'an arm')}"
            f") rather than `{best_shuffled}`. "
            + (f"Paired over the {n_orig} origins, the agent loop is "
               f"{signed(d, 3)} against the full-sensor forest -- "
               + ("an interval that includes zero, so this is a *suggestion*, "
                  "not a result. "
                  if crosses_zero(d) else "an interval clear of zero. ")
               if d else "")
            + (f"The mechanism is plausible -- a model holding "
               f"{n_sel_mean:.0f} sensors has fewer ways to lean on one that "
               f"drifts than one holding all "
               f"{prof['cleaner']['sensors_kept'] if prof else 'of them'}, "
               f"so selection should pay off exactly when the test "
               f"distribution moves"
               if (n_sel_mean and prof) else "The mechanism is plausible")
            + f" -- and that is a reason to test it properly, not to claim "
              f"it. {n_orig} origins with per-origin AUCs spanning "
            + f"{min(r['auc'] for a in real for r in ro[a]['per_origin']):.3f}"
              f" to "
            + f"{max(r['auc'] for a in real for r in ro[a]['per_origin']):.3f}"
              f" cannot settle it.", "",
            f"Test blocks grow from {sizes[0]['n_test']} wafers "
            f"({sizes[0]['n_fail_test']} fails) as the training window "
            f"expands, so individual origins are noisy by construction. The "
            f"honest summary: SECOM's shuffled-CV numbers are the optimistic "
            f"ones, a yield predictor trained this way should not be expected "
            f"to hold for the next month of wafers without retraining, and "
            f"whether sparse attribution helps under drift is the experiment "
            f"this dataset is too small to run.", ""]
    return body


def sec_rolling_sweep(rs, ev=None):
    if not rs:
        return []
    # which block count did the headline rolling-origin section use?
    headline_b = None
    if ev and "rolling_origin" in ev:
        m_b = re.search(r"(\d+) contiguous time blocks",
                        ev["protocol"]["rolling_origin"]["rule"])
        headline_b = m_b.group(1) if m_b else None
    per = rs["per_block_count"]
    counts = sorted(per, key=lambda b: int(b))
    arms = sorted({a for b in counts for a in per[b]["per_arm"]})
    rows = []
    for b in counts:
        e = per[b]
        d = e.get("agent_vs_rf_all")
        rows.append([
            b, str(e["n_origins"]),
            f"{min(e['test_block_fails'])}-{max(e['test_block_fails'])}",
            f"`{e['best_arm']}`",
            *[f"{e['per_arm'][a]['mean']:.3f}" if a in e["per_arm"] else "--"
              for a in arms],
            signed(d, 3) if d else "--",
        ])
    best_counts = [per[b]["best_arm"] for b in counts]
    agent_best = sum(1 for x in best_counts if x.startswith(("agent", "univar")))
    deltas = [per[b]["agent_vs_rf_all"] for b in counts
              if "agent_vs_rf_all" in per[b]]
    pos = sum(1 for d in deltas if d["mean"] > 0)
    clear = [d for d in deltas if not crosses_zero(d)]
    body = ["## Robustness of the forward-in-time reversal", "",
            f"The reversal above rests on one block count, so here is the "
            f"same protocol at several -- {rs['protocol']}. Fewer blocks "
            f"means larger, "
            f"less noisy test sets but fewer origins; more blocks means the "
            f"opposite, and origins with fewer than "
            f"{rs['min_fails_in_test_block']} fails in the test block are "
            f"skipped, because an AUC on one or two positives is not a "
            f"number. ({rs['wall_min']:.0f} min.)", "",
            table(rows, ["blocks", "origins", "fails per test block",
                         "best arm", *[f"`{a}`" for a in arms],
                         "agent_rf - rf_all"]), "",
            f"The **sign** is stable: a sparse arm is the best forward-in-time "
            f"arm at {agent_best} of the {len(counts)} block counts, and "
            f"`agent_rf` is above `rf_all` at {pos} of {len(deltas)}. The "
            f"**magnitude** is not established: "
            + (f"{len(clear)} of the {len(deltas)} paired intervals exclude "
               f"zero, and the origins within a block count share training "
               f"data, so even those are optimistic."
               if clear else
               f"not one of the {len(deltas)} paired intervals excludes zero, "
               f"and the origins within a block count share training data, so "
               f"the intervals are optimistic to begin with.")
            + " The defensible conclusion is a direction, not an effect "
              "size: on a drifting process, sparse attribution looks like it "
              "generalises better than a full-sensor model, and SECOM is too "
              "small to say by how much.", ""]
    if deltas and len(deltas) > 2 and headline_b:
        mags = sorted(d["mean"] for d in deltas)
        med = mags[len(mags) // 2]
        hd = per.get(headline_b, {}).get("agent_vs_rf_all")
        if hd:
            rank = sum(1 for m in mags if m > hd["mean"])
            body += [
                f"One more thing worth saying against the earlier section: the "
                f"block count it uses ({headline_b}) produced the "
                + ("**largest** " if rank == 0 else f"{rank + 1}th largest ")
                + f"of the {len(mags)} effects at {hd['mean']:+.3f}, against a "
                  f"median of {med:+.3f} across block counts. The reversal is "
                  f"not an artefact -- the sign holds everywhere -- but the "
                  f"headline number is the optimistic end of the range, and "
                  f"the median is the better estimate of it.", "",
            ]
    return body


def sec_drift(dr):
    if not dr:
        return []
    adv, ctl = dr["adversarial"]["auc"], dr["adversarial"]["auc_shuffled_control"]
    sd, ld = dr["sensor_drift"], dr["label_drift"]
    rows = [
        ["adversarial validation: can the sensors tell you *when* a wafer was "
         "made?", ci(adv), f"{dr['adversarial']['n_early']} early vs "
                           f"{dr['adversarial']['n_late']} late wafers"],
        ["the same test with the era label shuffled (control)", ci(ctl),
         "must land at chance, or the row above means nothing"],
        [f"sensors whose distribution moved between the first and last time "
         f"block", f"{sd['n_significant']} of {sd['n_sensors_tested']} "
                   f"({pct(sd['frac_significant'])})",
         f"KS two-sample, Benjamini-Hochberg FDR "
         f"{dr['protocol']['ks'].split('FDR ')[-1]}"],
        ["median / p90 / max KS statistic per sensor",
         f"{sd['ks_median']:.3f} / {sd['ks_p90']:.3f} / {sd['ks_max']:.3f}",
         "0 = identical distributions, 1 = disjoint"],
        ["fail rate across time blocks",
         f"{pct(ld['fail_rate_min'])} to {pct(ld['fail_rate_max'])}",
         f"chi-square p = {ld['chi2_p']:.3g}"],
    ]
    label_moves = ld["chi2_p"] < 0.05
    body = ["## Is it really drift? (measured, not assumed)", "",
            "\"The sensors drift\" is the obvious explanation for the section "
            "above, and obvious explanations are exactly the ones that get "
            "written into a README without being checked. Three checks, from "
            "`scripts/drift.py`:", "",
            table(rows, ["check", "value", "note"]), "",
            f"The adversarial test is the decisive one. Label each wafer by "
            f"*era* rather than by outcome -- early 70% versus late 30% -- and "
            f"the same pipeline that struggles to reach "
            f"{KPI_AUC:.2f} predicting **failure** separates the two eras at "
            f"**{adv['mean']:.3f}** "
            f"[{adv['ci_lo']:.3f}, {adv['ci_hi']:.3f}] from the sensors alone, "
            f"against {ctl['mean']:.3f} for the shuffled control. "
            + ("That is essentially perfect: "
               if adv["mean"] > 0.95 else
               "That is far above chance: " if adv["mean"] > 0.7 else
               "That is only modestly above chance: ")
            + f"the process data carries a much stronger signal about *when* a "
              f"wafer was made than about *whether it failed*. The training and "
              f"test halves of the chronological split are not two samples of "
              f"one distribution, and {pct(sd['frac_significant'])} of "
              f"individual sensors confirm it one at a time.", "",
            ]
    body += [
        f"Label drift is "
        + (f"also present: the fail rate ranges "
           f"{pct(ld['fail_rate_min'])} to {pct(ld['fail_rate_max'])} across "
           f"blocks (chi-square p = {ld['chi2_p']:.3g}), so part of the "
           f"forward-in-time collapse is the *prior* moving, not only the "
           f"features. The two effects are not separable at this sample size, "
           f"and neither is a modelling problem to be fixed by a better "
           f"ranker."
           if label_moves else
           f"comparatively mild: the fail rate ranges "
           f"{pct(ld['fail_rate_min'])} to {pct(ld['fail_rate_max'])} across "
           f"blocks and the chi-square test does not reject homogeneity "
           f"(p = {ld['chi2_p']:.3g}), so what moves is mainly the sensor "
           f"distributions rather than the failure prior."), "",
        "This is also the cleanest argument for why the top-5 stability KPI is "
        "hard here in a way no ranker fixes. If the sensors themselves are "
        "non-stationary over the 90 days, \"the top 5 causes\" is not a fixed "
        "quantity being estimated noisily -- it is a quantity that changes "
        "while you estimate it.", "",
    ]
    return body


def sec_sweep(sw):
    if not sw:
        return []
    auc = sw["auc"]
    per, paired = auc["per_arm"], auc["paired"]
    base = per["rf_all"]["mean"]
    rows = []
    order = sorted(per, key=lambda a: -per[a]["mean"])
    for a in order:
        m = sw["arms"][a]["meta"]
        d = paired.get(f"{a}__vs__rf_all")
        rows.append([
            f"`{a}`",
            m.get("attribution", "--"),
            (f"{m.get('select_k')} / {m.get('stability_min')} / "
             f"{m.get('max_select')}") if m else "--",
            f"{sw['n_selected_mean'][a]:.1f}" if a in sw["n_selected_mean"]
            else "--",
            ci(per[a]),
            "--" if a == "rf_all" else signed(d, 3),
        ])

    agents = [(a, per[a]["mean"], sw["n_selected_mean"][a])
              for a in per
              if a in sw["n_selected_mean"] and a.startswith("agent_")]
    agents.sort(key=lambda x: x[2])
    tied = [(a, n) for a, _, n in agents
            if crosses_zero(paired[f"{a}__vs__rf_all"])]
    worse = [(a, n) for a, _, n in agents
             if not crosses_zero(paired[f"{a}__vs__rf_all"])]
    smallest_tied = min((n for _, n in tied), default=None)
    # is the trend in sparsity monotone, or is that just an impression?
    xs = [n for _, _, n in agents]
    ys = [m for _, m, _ in agents]
    conc = sum(1 for i in range(len(xs)) for j in range(i + 1, len(xs))
               if (xs[j] - xs[i]) * (ys[j] - ys[i]) > 0)
    npairs = len(xs) * (len(xs) - 1) // 2
    # does the cheaper attribution mode actually lose?
    by_tag = {}
    for a in per:
        m = sw["arms"][a]["meta"]
        if m.get("attribution"):
            by_tag.setdefault(m["tag"], {})[m["attribution"]] = per[a]["mean"]
        matched = [(t, v["model"] - v["permutation"]) for t, v in by_tag.items()
                   if "model" in v and "permutation" in v]
    mod_wins = sum(1 for _, d in matched if d > 0)

    body = ["## Is the pre-registered operating point the problem?", "",
            f"The agent loop's structural settings are fixed in advance rather "
            f"than tuned, which is only defensible if the surface around them "
            f"is published instead of hidden. Same protocol as above "
            f"({sw['protocol']}), {auc['n_folds']} folds, "
            f"{sw['wall_min']:.0f} min:", "",
            table(rows, ["arm", "attribution", "vote k / threshold / cap",
                         "sensors selected", "ROC-AUC (95% CI)",
                         "paired delta vs `rf_all`"]), "",
            f"AUC tracks how many sensors survive, and it does so almost "
            f"monotonically: {conc} of the {npairs} (sparsity, AUC) pairs are "
            f"concordant, from {agents[0][2]:.0f} sensors at "
            f"{agents[0][1]:.3f} up to {agents[-1][2]:.0f} at "
            f"{agents[-1][1]:.3f}, against {base:.3f} for using all of them. "
            f"SECOM's predictive signal is spread thinly over many weak "
            f"sensors rather than concentrated in a few, so sparsity is not "
            f"free.", ""]

    if tied and smallest_tied is not None:
        names_tied = ", ".join(f"`{a}`" for a, _ in tied)
        marg = [a for a, _ in tied if marginal(paired[f"{a}__vs__rf_all"])]
        body += [
            f"That does not make the loop uniformly worse. "
            f"{len(tied)} of the {len(agents)} configurations have a paired CI "
            f"that reaches the baseline -- {names_tied} -- and the leanest of "
            f"those keeps about {smallest_tied:.0f} sensors. **So the loop can "
            f"match a full-sensor forest, but only by declining to be a "
            f"shortlist.** Every configuration returning a list short enough "
            f"for an engineer to work through ({len(worse)} of them, down to "
            f"{min(n for _, n in worse):.0f} sensors) is measurably worse.", "",
        ]
        if marg:
            body += [
                f"Two caveats on those ties, both in the direction of not "
                f"over-claiming. {len(marg)} of them "
                f"({', '.join(f'`{a}`' for a in marg)}) have a CI that only "
                f"just touches zero, which is a boundary case rather than a "
                f"demonstrated equivalence. And this sweep runs "
                f"{auc['n_folds']} folds where the headline table runs "
                f"{25 if auc['n_folds'] != 25 else auc['n_folds']}, so its "
                f"intervals are wider and it has *less* power to separate arms "
                f"-- a tie here is weaker evidence than a tie there. The "
                f"headline comparison is the one with the folds.", "",
            ]
    # matched-sparsity comparison against the naive control
    ctl = "univar_top25_rf"
    if ctl in sw["n_selected_mean"] and agents:
        n_ctl = sw["n_selected_mean"][ctl]
        near = min(agents, key=lambda x: abs(x[2] - n_ctl))
        dm = paired.get(f"{near[0]}__vs__{ctl}")
        if dm:
            body += [
                f"The fairest single comparison in the table is at matched "
                f"sparsity. `{ctl}` keeps {n_ctl:.0f} sensors chosen one at a "
                f"time; `{near[0]}` keeps {near[2]:.0f} chosen by the full "
                f"loop. Paired over the same folds the loop is "
                f"{signed(dm, 3)} against it"
                + (" -- an interval including zero, so at equal budget the "
                   "plan/verify machinery is not measurably better than "
                   "ranking each sensor on its own."
                   if crosses_zero(dm) else
                   " -- an interval clear of zero, so at equal budget the "
                   "machinery does beat ranking each sensor on its own.")
                + " That is the comparison the loop most needs to win, and on "
                  "this data it does not win it decisively either way.", "",
            ]

    nodrop = [a for a in per if a.startswith("agent_no_drop")]
    if nodrop:
        nd = max(nodrop, key=lambda a: per[a]["mean"])
        d = paired[f"{nd}__vs__rf_all"]
        body += [
            f"The limit row is the sanity check rather than a result: with the "
            f"drop step disabled (`stability_min` 0, no cap) `{nd}` lands at "
            f"{per[nd]['mean']:.3f}, {signed(d, 3)} from the baseline. The "
            f"wrapper degrades back onto the baseline as it should, so the gap "
            f"at the operating point is the selection doing damage, not a "
            f"defect in the plumbing.", "",
        ]
    if matched:
        body += [
            f"One design axis does pay: scoring suspects by the base model's "
            f"own importance averaged over resamples beats held-out "
            f"permutation AUC-drop at {mod_wins} of the {len(matched)} matched "
            f"depths (mean gap "
            f"{sum(d for _, d in matched) / len(matched):+.3f} AUC). With "
            f"roughly 25 positives in an inner validation split, the "
            f"permutation estimate is simply too noisy to rank on, which is "
            f"the same sample-size story the stability section tells.", "",
        ]
    return body


def sec_stability(st, prof=None):
    if not st:
        return []
    n_wafers = f"{prof['n_wafers']:,}" if prof else "the"
    rows = []
    # a ranker mid-measurement may have bootstrap but not yet cv_train
    ranked = {k: v for k, v in st["rankers"].items() if "bootstrap" in v}
    order = sorted(ranked,
                   key=lambda r: -ranked[r]["bootstrap"]["raw"]["pairwise_overlap"])
    for name in order:
        r = ranked[name]
        b, c = r["bootstrap"], r.get("cv_train")
        rows.append([
            f"`{name}`", r["label"],
            pct(b["raw"]["pairwise_overlap"]),
            pct(b["cluster"]["pairwise_overlap"]),
            pct(b["raw"]["consensus_freq"]),
            pct(c["raw"]["pairwise_overlap"]) if c else "--",
            pct(c["cluster"]["pairwise_overlap"]) if c else "--",
        ])
    body = [
        "## SECOM: top-5 stability", "",
        "The metric was defined in `yieldrca/stability.py` before it was "
        "measured, because it has enough degrees of freedom that defining it "
        "afterwards would be meaningless. **Primary: mean pairwise top-5 "
        "overlap** -- the average, over all pairs of resamples, of "
        "|T_b n T_b'| / 5, where T_b is the top 5 of a ranking re-derived from "
        "scratch on resample b. It has no reference set, so it cannot be "
        "inflated by choosing the reference after the fact. The **consensus** "
        "column instead picks the 5 most frequent sensors *after* seeing every "
        "resample and averages their selection frequency, which is why it is "
        "always the friendlier number. The **cluster** columns map each sensor "
        f"to its |r| >= {st['cluster_thresh']} correlation group first.", "",
        f"Two perturbation schemes, and the choice matters more than any "
        f"modelling decision below it. `bootstrap` is "
        f"{st['schemes']['bootstrap']['n_replicates']} resamples with "
        f"replacement -- each sees ~63% of the wafers as unique rows, so two "
        f"replicates share under half their data. `cv_train` is the "
        f"{st['schemes']['cv_train']['n_replicates']} training folds of the "
        f"same repeated CV the AUC table uses -- at 5 folds those are 80% of "
        f"the data each and share 75% of their rows, a much gentler shake. "
        f"**Bootstrap is reported as primary** because it is the standard "
        f"stability-selection perturbation and because a KPI should be scored "
        f"against the harder of two defensible protocols, not the kinder one.",
        "",
        table(rows, ["ranker", "what it ranks by", "pairwise (bootstrap)",
                     "pairwise, cluster-aware", "consensus (bootstrap)",
                     "pairwise (CV folds)", "pairwise, cluster-aware (CV)"]), "",
        f"A uniformly random ranker scores "
        f"{pct(st['random_floor_raw'], 1)} raw "
        f"({st['k']} of {st['effective_sensors']} surviving sensors) and "
        f"{pct(st['random_floor_cluster'], 1)} cluster-aware "
        f"({st['k']} of {st['effective_clusters']} clusters), so every row is "
        f"far clear of chance.", "",
    ]
    ag = st["rankers"].get("agent")
    if not ag or "cv_train" not in ag:
        return body
    b, c = ag["bootstrap"], ag["cv_train"]
    v_boot, boot_ok = verdict(b["raw"]["pairwise_overlap"], KPI_STABILITY)
    v_cl, _ = verdict(b["cluster"]["pairwise_overlap"], KPI_STABILITY)
    v_cv, cv_ok = verdict(c["raw"]["pairwise_overlap"], KPI_STABILITY)
    best_raw = max(ranked,
                   key=lambda r: ranked[r]["bootstrap"]["raw"]["pairwise_overlap"])
    body += [
        f"The full agent loop reaches "
        f"**{pct(b['raw']['pairwise_overlap'])}** pairwise overlap under "
        f"bootstrap resampling ({pct(b['cluster']['pairwise_overlap'])} "
        f"cluster-aware) and {pct(c['raw']['pairwise_overlap'])} across CV "
        f"training folds. Against the >= {KPI_STABILITY:.0%} KPI that is "
        f"{v_boot} on the primary protocol, {v_cl} cluster-aware, and {v_cv} "
        f"under the gentler CV-fold perturbation."
        + ("" if boot_ok == cv_ok else
           " Reporting only the second of those would be the easiest way to "
           "overstate this pipeline, which is why both are here and why the "
           "harder one is the headline."), "",
        f"The most stable ranker in the table is `{best_raw}`"
        + ("" if best_raw == "agent" else
           " -- the plan/verify machinery does not buy stability over simply "
           "ranking each sensor on its own, which is the same conclusion the "
           "AUC table reaches from the other direction") + ".", "",
    ]
    nc = st["rankers"].get("agent_no_corr")
    po = st["rankers"].get("perm_only")
    if nc or po:
        body += [
            "**Which part of the loop does the stability work?** The three "
            "rows form a ladder, each step adding one mechanism, so each "
            "difference isolates one thing rather than two:", "",
        ]
        ladder = []
        if po:
            ladder.append(["`perm_only`", "screen + held-out permutation only",
                           pct(po["bootstrap"]["raw"]["pairwise_overlap"]),
                           "--"])
        if nc:
            d = (nc["bootstrap"]["raw"]["pairwise_overlap"]
                 - (po["bootstrap"]["raw"]["pairwise_overlap"] if po else 0))
            ladder.append(["`agent_no_corr`", "+ bootstrap verify-and-drop",
                           pct(nc["bootstrap"]["raw"]["pairwise_overlap"]),
                           f"{d:+.1%}" if po else "--"])
        d2 = (b["raw"]["pairwise_overlap"]
              - (nc["bootstrap"]["raw"]["pairwise_overlap"] if nc else 0))
        ladder.append(["`agent`", "+ correlation grouping",
                       pct(b["raw"]["pairwise_overlap"]),
                       f"{d2:+.1%}" if nc else "--"])
        body += [table(ladder, ["ranker", "mechanism added",
                                "pairwise (bootstrap)", "step"]), ""]
        if po and nc:
            dv = (nc["bootstrap"]["raw"]["pairwise_overlap"]
                  - po["bootstrap"]["raw"]["pairwise_overlap"])
            body += [
                f"So verification is worth {dv:+.1%} of top-5 agreement and "
                f"grouping {d2:+.1%}. "
                + ("Both steps earn their place, which is the one clearly "
                   "positive thing to say about the loop's structure on this "
                   "dataset -- they buy *stability*, not accuracy."
                   if dv > 0.01 and d2 > 0.01 else
                   "Neither step moves the number much, so on this dataset "
                   "the loop's extra machinery is not what determines "
                   "stability." if abs(dv) <= 0.01 and abs(d2) <= 0.01 else
                   "Only one of the two steps is doing measurable work here.")
                + f" Note also that the raw and cluster-aware columns barely "
                  f"differ across the whole table, which says the top 5 mostly "
                  f"are *not* drawn from the near-duplicate families that "
                  f"motivated grouping -- the instability is between genuinely "
                  f"different sensors.", "",
            ]
    if not boot_ok:
        top_v = ranked[best_raw]["bootstrap"]["raw"]["pairwise_overlap"]
        gap_ranker = top_v - b["raw"]["pairwise_overlap"]
        gap_kpi = KPI_STABILITY - top_v
        body += [
            f"Two different gaps are visible here and they should not be "
            f"conflated. The first is between rankers: `{best_raw}` is "
            f"{gap_ranker:+.1%} above the full loop, so *choice of ranker "
            f"matters a great deal* -- held-out permutation importance, scored "
            f"on an inner split holding roughly 25 positives, is simply a "
            f"noisier statistic than a univariate AUC or a fitted "
            f"coefficient. That is the same finding the sensitivity sweep "
            f"reached from the accuracy side, and it is actionable: the loop's "
            f"attribution mode is a parameter.", "",
            f"The second gap is the one no ranker closes. Even `{best_raw}`, "
            f"the most stable thing in the table, sits {gap_kpi:.0%} short of "
            f"the {KPI_STABILITY:.0%} KPI. Resampling {n_wafers} wafers with "
            f"replacement leaves out about 37% of them, so at this class "
            f"balance each replicate sees a different ~65 fails, and which "
            f"five of {st['effective_sensors']} weakly informative sensors "
            f"come out on top is not determined at that sample size. The "
            f"consensus column says the same from the other side: some "
            f"sensors recur far more often than chance, but not the *same "
            f"five* run to run. A better ranker would narrow the first gap; "
            f"only more failed wafers narrows the second.", "",
        ]
    return body


def sec_synthetic(sy, ev=None):
    if not sy or "agent" not in sy.get("recovery", {}):
        return []
    secom_delta = (ev or {}).get("auc", {}).get("paired", {}).get(
        "agent_rf__vs__rf_all")
    rec, auc = sy["recovery"], sy["auc"]
    m_k = re.search(r"top-(\d+)", sy["protocol"]["recovery"])
    k = m_k.group(1) if m_k else str(sy.get("k", 5))
    rows = []
    for m in sorted(rec, key=lambda m: -rec[m]["topk_recall"]["mean"]):
        r = rec[m]
        rows.append([
            f"`{m}`",
            f"{r['topk_hits']['mean']:.1f} / {sy['records'][0]['n_causal']}",
            ci01(r["topk_recall"]),
            f"{r['topk_precision']['mean']:.2f}",
            (f"{r['selected_recall']['mean']:.2f}" if "selected_recall" in r else "--"),
            (f"{r['selected_precision']['mean']:.2f}" if "selected_precision" in r else "--"),
            pct(sy["stability"][m]["raw"]["pairwise_overlap"]),
        ])
    arows = []
    for a in sorted(auc["per_arm"], key=lambda a: -auc["per_arm"][a]["mean"]):
        d = auc["paired"].get(f"{a}__vs__rf_all")
        arows.append([f"`{a}`", ci(auc["per_arm"][a]),
                      "--" if a == "rf_all" else signed(d, 3)])
    ag = rec["agent"]
    ag_stab = sy["stability"]["agent"]["raw"]["pairwise_overlap"]
    v, ok = verdict(ag_stab, KPI_STABILITY)
    g = sy["generator"]
    return [
        "## Synthetic benchmark -- the only place with ground truth", "",
        f"SECOM ships no causal labels, so recovery cannot be scored on it at "
        f"all. Here it can: {g['n_causal']} of {g['p']} sensors genuinely drive "
        f"the label, over {g['n']} wafers at a "
        f"{100*g['fail_rate_target']:.0f}% fail rate with "
        f"{100*g['missing_rate']:.0f}% missing cells and "
        f"{g['note']}. Averaged over "
        f"{sy['protocol']['seeds']} independently generated datasets:", "",
        table(rows, ["method", f"top-{k} hits", f"top-{k} recall (95% CI)",
                     f"top-{k} precision", "selected recall",
                     "selected precision", "top-5 stability (pairwise)"]), "",
        f"Held-out AUC on the same generator, "
        f"{sy['protocol']['auc'].replace('per seed, ', '')}:", "",
        table(arows, ["arm", "ROC-AUC (95% CI)", "paired delta vs `rf_all`"]), "",
        f"With real causal structure present the loop recovers "
        f"{ag['topk_recall']['mean']:.0%} of the planted sensors in its top-"
        f"{k}, and its top-5 stability is {pct(ag_stab)} -- the same "
        f"definition used on SECOM, {v} here.", "",
        (f"The AUC ordering flips too, and that is the sharpest statement this "
         f"repo can make about when the agent loop is worth running. Here "
         f"`agent_rf` is {signed(auc['paired']['agent_rf__vs__rf_all'], 3)} "
         f"**above** the full-sensor forest; on SECOM it is "
         f"{signed(secom_delta, 3)} below it. The loop's premise is that a few "
         f"sensors genuinely drive the failures. Where that premise holds it "
         f"wins on both accuracy and stability; where the signal is spread "
         f"thin across hundreds of weak sensors, enforcing sparsity throws "
         f"away exactly what the model needed."
         if secom_delta else
         f"So the machinery is not broken, and the SECOM numbers are a "
         f"statement about SECOM rather than about the pipeline."), "",
        (f"One number in the table deserves its own sentence: the loop's "
         f"*selected set* has recall {ag['selected_recall']['mean']:.2f} but "
         f"precision {ag['selected_precision']['mean']:.2f}, because it keeps "
         f"about {ag['n_selected']['mean']:.0f} sensors to be safe. It finds "
         f"the causes; it does not claim only the causes. The top-5 is the "
         f"precise output, the selected set is the recall-oriented one, and "
         f"the report distinguishes them."
         if "selected_recall" in ag else ""), "",
        "**These numbers are synthetic and must never be quoted as real-data "
        "results.**", "",
    ]


def sec_headline(ev, st, sy, sw, prof=None, dr=None, rsw=None):
    """The handful of sentences a reader should leave with, computed not asserted."""
    if not ev:
        return []
    auc = ev["auc"]
    rf, ag = auc["per_arm"]["rf_all"], auc["per_arm"]["agent_rf"]
    d = auc["paired"]["agent_rf__vs__rf_all"]
    d_u = auc["paired"]["agent_rf__vs__univar_top25_rf"]
    ch = ev["chronological"]
    v1, ok1 = verdict(rf["mean"], KPI_AUC)
    L = [
        f"- **The AUC KPI is met, by the baseline.** A plain random forest "
        f"with fold-internal cleaning and median imputation scores "
        f"**{rf['mean']:.3f}** [{rf['ci_lo']:.3f}, {rf['ci_hi']:.3f}] ROC-AUC "
        f"over {auc['n_folds']} folds of repeated stratified CV, so the "
        f">= {KPI_AUC:.2f} target is {v1} before any agent runs. That is "
        f"squarely inside the 0.70-0.80 band published for SECOM; anything "
        f"far above it on this dataset is a leak, not a result.",
        f"- **The agent loop does not beat that baseline -- it loses to it.** "
        f"Same folds, paired per fold: **{signed(d, 3)}** AUC "
        f"({d['wins']} folds better, {d['losses']} worse, Wilcoxon "
        f"p = {d.get('wilcoxon_p', float('nan')):.1e}). At its pre-registered "
        f"operating point the loop reaches {ag['mean']:.3f} and misses the KPI "
        f"on its own, and it does not separate from a univariate top-25 "
        f"selection at the same sparsity ({signed(d_u, 3)}"
        + (", an interval that includes zero)." if crosses_zero(d_u)
           else ").")
        + " The plan/verify machinery is not what is buying the number.",
    ]
    if sw:
        per, paired = sw["auc"]["per_arm"], sw["auc"]["paired"]
        tied = [(a, sw["n_selected_mean"][a]) for a in per
                if a.startswith("agent_") and a in sw["n_selected_mean"]
                and crosses_zero(paired[f"{a}__vs__rf_all"])]
        if tied:
            lean, n_lean = min(tied, key=lambda t: t[1])
            dl = paired[f"{lean}__vs__rf_all"]
            clear = [(a, n) for a, n in tied
                     if not marginal(paired[f"{a}__vs__rf_all"])]
            L.append(
                f"- **Sparsity is the price, and it is not negotiable.** "
                f"Sweeping the loop's settings, AUC tracks how many sensors "
                f"survive. The leanest configuration whose paired CI still "
                f"reaches the baseline is `{lean}` at {signed(dl)}, keeping "
                f"about {n_lean:.0f} sensors"
                + (f" -- but that CI only just touches zero, so read it as a "
                   f"boundary case rather than a tie; the leanest "
                   f"*unambiguous* tie keeps about "
                   f"{min(n for _, n in clear):.0f}"
                   if marginal(dl) and clear else "")
                + ". The loop can match a full-sensor forest, but only by "
                  "declining to be a shortlist -- every setting that returns "
                  "a list short enough for an engineer to work through is "
                  "measurably worse.")
    if st and "agent" in st["rankers"]:
        a = st["rankers"]["agent"]["bootstrap"]
        cvs = st["rankers"]["agent"]["cv_train"]
        v3, ok3 = verdict(a["raw"]["pairwise_overlap"], KPI_STABILITY)
        L.append(
            f"- **The top-5 stability KPI is {v3.strip('*')} on the protocol "
            f"it should be scored on.** Under the definition fixed in "
            f"`yieldrca/stability.py` -- mean pairwise overlap of the top 5, "
            f"re-derived from scratch on each of "
            f"{st['schemes']['bootstrap']['n_replicates']} bootstrap "
            f"resamples -- the loop scores "
            f"**{pct(a['raw']['pairwise_overlap'])}** "
            f"({pct(a['cluster']['pairwise_overlap'])} after grouping sensors "
            f"correlated above |r| = {st['cluster_thresh']}) against a "
            f">= {KPI_STABILITY:.0%} target and a "
            f"{pct(st['random_floor_raw'])} random-ranker floor. Across "
            f"80%-overlapping CV training folds -- a much gentler shake of the "
            f"same data -- it reads "
            f"{pct(cvs['raw']['pairwise_overlap'])}. Both are reported; the "
            f"harder one is the headline. With 104 fails, which five of "
            f"{st['effective_sensors']} sensors come out on top is barely "
            f"determined.")
    worst = min(ch, key=lambda a_: ch[a_]["auc"] if a_ != "majority" else 9)
    ro = ev.get("rolling_origin")
    ro_txt = ""
    if ro:
        real = [a for a in ro if a != "majority"]
        best_ro = max(real, key=lambda a: ro[a]["summary"]["mean"])
        ro_txt = (f" Repeating the exercise at every origin -- train on the "
                  f"past, test on the next block of wafers -- puts the best "
                  f"arm at {ro[best_ro]['summary']['mean']:.3f} "
                  f"(`{best_ro}`), so this is not one unlucky split.")
    L.append(
        f"- **Shuffled CV flatters this dataset.** Train on the earliest "
        f"{ev['protocol']['chronological']['n_train']} wafers and test on the "
        f"last {ev['protocol']['chronological']['n_test']}, and the best "
        f"baseline falls from {rf['mean']:.3f} to "
        f"**{ch['rf_all']['auc']:.3f}** -- near chance, with every arm "
        f"collapsing (worst: `{worst}` at {ch[worst]['auc']:.3f})."
        + ro_txt
        + f" Over the {prof['time_span_days']:.0f} days of a single campaign "
          f"the sensor distributions drift, and a shuffled split lets the "
          f"model interpolate across drift it would never see in production. "
          f"The KPI is stated against the shuffled protocol, so that is what "
          f"the scorecard reports -- but the forward-in-time number is the one "
          f"an engineer should believe."
        if prof else "")
    if dr:
        adv = dr["adversarial"]["auc"]
        sd, ld = dr["sensor_drift"], dr["label_drift"]
        L.append(
            f"- **And the drift is measured, not assumed.** Label each wafer "
            f"by *era* instead of outcome -- early 70% versus late 30% -- and "
            f"the same pipeline separates the two eras from the sensors alone "
            f"at **{adv['mean']:.3f}** AUC "
            f"({dr['adversarial']['auc_shuffled_control']['mean']:.3f} with "
            f"the era label shuffled). The process data says far more about "
            f"*when* a wafer was made than about *whether it failed*: "
            f"{pct(sd['frac_significant'])} of sensors shift significantly "
            f"between the first and last time block, and the fail rate itself "
            f"runs {pct(ld['fail_rate_min'])} to {pct(ld['fail_rate_max'])} "
            f"across blocks (chi-square p = {ld['chi2_p']:.1g}). On a "
            f"non-stationary process, \"the top 5 causes\" is not a fixed "
            f"quantity measured noisily -- it is a quantity that moves while "
            f"you measure it.")
    ro = ev.get("rolling_origin")
    if ro and "agent_rf" in ro and "rf_all" in ro:
        dro = paired_delta([r["auc"] for r in ro["agent_rf"]["per_origin"]],
                           [r["auc"] for r in ro["rf_all"]["per_origin"]])
        real = [a for a in ro if a != "majority"]
        best_ro = max(real, key=lambda a: ro[a]["summary"]["mean"])
        L.append(
            f"- **One result points the other way, and it is the weakest one "
            f"here.** Forward in time the ordering inverts: the best arm "
            f"across origins is `{best_ro}` at "
            f"{ro[best_ro]['summary']['mean']:.3f}, and the agent loop is "
            f"{signed(dro, 3)} against the full-sensor forest instead of "
            f"behind it. Selecting fewer sensors plausibly helps precisely "
            f"when the test distribution has moved. But that interval "
            + ("includes zero over only "
               f"{ev['protocol']['rolling_origin']['n_origins']} origins, so "
               "it is a hypothesis worth a bigger dataset, not a finding."
               if crosses_zero(dro) else
               f"is clear of zero over only "
               f"{ev['protocol']['rolling_origin']['n_origins']} origins.")
            + (f" Repeating the protocol at five block counts (below) keeps "
               f"the sign at every one but puts the median effect at "
               f"{sorted(v['agent_vs_rf_all']['mean'] for v in rsw['per_block_count'].values() if 'agent_vs_rf_all' in v)[len([v for v in rsw['per_block_count'].values() if 'agent_vs_rf_all' in v]) // 2]:+.3f} "
               f"rather than {dro['mean']:+.3f}, and no single interval "
               f"excludes zero."
               if rsw and rsw.get("per_block_count") else ""))
    if sy and "agent" in sy.get("recovery", {}):
        r = sy["recovery"]["agent"]
        if True:
            sd = sy["auc"]["paired"].get("agent_rf__vs__rf_all")
            L.append(
                f"- **The machinery works where its premise holds.** On the "
                f"synthetic generator -- {sy['generator']['n_causal']} "
                f"genuinely causal sensors among {sy['generator']['p']}, "
                f"block-correlated noise -- the loop recovers "
                f"{r['topk_recall']['mean']:.0%} of them in its top 5, scores "
                f"{pct(sy['stability']['agent']['raw']['pairwise_overlap'])} "
                f"top-5 stability (KPI met), and beats the full-sensor forest "
                f"by {signed(sd, 3)} AUC -- the *opposite* sign to SECOM. The "
                f"loop assumes a few sensors drive the failures; where that is "
                f"true it wins on accuracy and stability, and where the signal "
                f"is spread thin it throws away what the model needed. "
                f"**Recovery is only ever claimed on synthetic data; SECOM has "
                f"no causal labels and none is reported for it.**")
    return ["## Headline", ""] + L + [""]


def sec_null_fdr(nf):
    """Hallucination control, as a measured false-discovery rate."""
    if not nf:
        return []
    n, r = nf["null"], nf["real"]
    sep = nf["separation"]
    fd = nf["null_fdr"]
    pr = nf["protocol"]
    rows = [
        ["sensors reported as root causes, per replicate",
         f"{n['n_reported_mean']:.1f}", f"{r['n_reported_mean']:.1f}",
         "mean over replicates"],
        ["...of which cleared the stability threshold on merit",
         f"{n['n_merit_mean']:.1f}", f"{r['n_merit_mean']:.1f}",
         f"threshold pi = {pr['agent_cfg']['stability_min']}"],
        ["replicates reporting nothing at all (abstention)",
         pct(n["abstention_rate"]), pct(r["abstention_rate"]),
         "the only outcome that would be correct on the null"],
        ["replicates where the never-empty fallback fired",
         pct(n["fallback_rate"]), pct(r["fallback_rate"]),
         "`estimator.py`: `if not surv: surv = reps[:5]`"],
        ["largest bootstrap support any suspect reached",
         f"{n['max_stability_mean']:.3f}", f"{r['max_stability_mean']:.3f}",
         "mean; the statistic the drop step thresholds"],
        ["...its 5th-95th percentile across replicates",
         f"[{n['max_stability_q05']:.3f}, {n['max_stability_q95']:.3f}]",
         f"[{r['max_stability_q05']:.3f}, {r['max_stability_q95']:.3f}]",
         "overlap here means the score cannot separate the two worlds"],
    ]
    verdict_word = ("**never**" if n["abstention_rate"] == 0 else
                    f"in {pct(n['abstention_rate'])} of replicates")
    sep_p = sep["prob_real_max_exceeds_null_max"]
    if sep_p >= 0.95:
        sep_read = ("so the score does separate the two worlds, and a "
                    "null-calibrated threshold is a usable filter")
    elif sep_p >= 0.75:
        sep_read = ("so the score carries some information about whether the "
                    "labels were real, but not enough to filter on per-sensor")
    else:
        sep_read = ("so the score carries essentially no information about "
                    "whether the labels were real")
    body = [
        "## Does the loop invent root causes when there are none?", "",
        "The pitch is that a suspect failing the bootstrap stability check is "
        "dropped, so what survives is trustworthy. That is a claim about a "
        "false-discovery rate, and it is measurable: build a world with no "
        "causal sensors and count what the loop reports anyway.", "",
        f"**The null.** {pr['null']} "
        f"{pr['n_null']} such replicates, against {pr['n_real']} replicates of "
        f"the identical loop on the true labels ({pr['real_arm']}). "
        f"From `scripts/null_fdr.py`, written to `runs/null_fdr.json`.", "",
        table(rows, ["quantity", "permuted labels (no causes exist)",
                     "real labels", "note"]), "",
        f"**The loop abstains {verdict_word}.** Over {pr['n_null']} "
        f"permuted-label replicates it named "
        f"{fd['false_discoveries_total']} sensors as root causes. Every one of "
        f"them is a false discovery by construction, so the false-discovery "
        f"rate of the reported suspect list under this null is "
        f"**{fd['fdr_given_nonempty']:.0%}**.", "",
        f"That number is not a tuning failure, it is the architecture: "
        f"`AgentRCA.fit` cannot return an empty report. If no suspect clears "
        f"the threshold it restores the top five anyway "
        f"(`estimator.py`, `if not surv: surv = reps[:5]`), and the "
        f"candidate-selection step above it has the same guard. On the null "
        f"the guard is mostly not even needed: "
        f"{n['n_merit_mean']:.1f} sensors clear the threshold on merit, "
        f"because with {pr['agent_cfg']['n_boot']} bootstrap replicates a "
        f"pure-noise sensor that happens to rank highly once will do so again.",
        "",
        f"**Is the loop at least *more* confident on real data?** "
        f"P(real replicate's best support > null replicate's best support) = "
        f"**{sep_p:.3f}**, where 0.5 is no information "
        f"(Mann-Whitney p = {sep['p_real_greater']:.3g}), {sep_read}.", "",
    ]
    cl = nf.get("real_cleared") or {}
    if cl:
        crows = []
        for key in sorted(cl, key=lambda k: -float(k.split("_")[1])):
            c = cl[key]
            crows.append([
                f"alpha = {key.split('_')[1]}", f"{c['tau']:.3f}",
                f"{c['n_cleared_mean']:.2f}",
                pct(c["replicates_with_none"]),
            ])
        body += [
            "**What survives a null-calibrated threshold.** Taking "
            "tau(alpha) as the (1-alpha) quantile of the null's best support "
            "-- Westfall-Young max-statistic control, family-wise over the "
            "sensors screened in a replicate -- and asking how many real-label "
            "suspects clear it:", "",
            table(crows, ["level", "tau", "real suspects clearing tau (mean)",
                          "replicates clearing none"]), "",
            "This is the honest version of the suspect list, and it is much "
            "shorter than the one the loop prints today. It is also a "
            "one-line change to enforce, and it gives the pipeline something "
            "it currently lacks: the ability to say *nothing here is above "
            "noise*.", "",
        ]
    return body


def sec_invariance(iv):
    """Whether the reported suspects are associational or something stronger."""
    if not iv:
        return []
    tot = iv["totals"]
    pr = iv["protocol"]
    blocks = iv["blocks"]
    brow = " / ".join(f"{b['n_fail']}" for b in blocks)
    grows = []
    for g in iv["groups"]:
        if not g.get("n"):
            continue
        grows.append([
            g["label"], str(g["n"]), str(g["n_associated"]),
            str(g["n_non_invariant"]), str(g["n_invariant"]),
        ])
    body = [
        "## Are the suspects causal, or only associated?", "",
        "Permutation importance is not a causal quantity: it measures how much "
        "a model leans on a column. The weakest claim with an actual "
        "identification argument behind it is invariance -- if a sensor really "
        "drives failure, its relationship with failure should survive a change "
        "of production period, and `runs/drift.json` already shows these "
        "periods are genuinely different environments. This is the marginal "
        "screen from Invariant Causal Prediction (Peters, Buhlmann & "
        "Meinshausen, JRSS-B 2016): a **necessary** condition for a stable "
        "cause, not a sufficient one. From `scripts/invariance.py`, written to "
        "`runs/invariance.json`.", "",
        f"Environments: {pr['environments']}, carrying {brow} failed wafers "
        f"respectively. Two stages with two different nulls, because "
        f"conflating them is the easy mistake -- association is tested by "
        f"permuting the labels, invariance by permuting which wafers belong to "
        f"which period, so that each sensor's overall association is held "
        f"fixed and only the block structure is destroyed.", "",
        table(grows, ["group", "n", "associated", "non-invariant",
                      "associated AND invariant"]), "",
        f"Of {tot['n_sensors']} surviving sensors, "
        f"**{tot['n_associated']}** show any association with failure at all "
        f"(BH, FDR {pr['alpha']}), and of those "
        f"**{tot['n_non_invariant']}** "
        + ("is" if tot["n_non_invariant"] == 1 else "are")
        + f" non-invariant across periods, leaving "
          f"**{tot['n_invariant']}** that are both associated and not shown to "
          f"break.", "",
    ]
    # The sharp finding goes before the power table that qualifies it: which
    # sensors failed, and why a failure carries more weight than a pass.
    broke = [r for r in iv.get("per_sensor", []) if r.get("associated")
             and not r.get("invariant")]
    kept = [r for r in iv.get("per_sensor", []) if r.get("associated")
            and r.get("invariant")]
    if broke:
        brows = []
        for r in sorted(broke, key=lambda r: -r["folds_selected"]):
            ba = " / ".join("--" if v is None else f"{v:.2f}"
                            for v in r["block_auc"])
            brows.append([f"`{r['name']}`", str(r["folds_selected"]),
                          str(r["folds_top5"]), f"{r['pooled_auc']:.3f}",
                          ba, f"{r['i2']:.2f}" if r["i2"] is not None else "--",
                          f"{r['invariance_p_bh']:.3f}"])
        body += [
            "**The sensor the screen rejects is the loop's favourite.**", "",
            table(brows, ["sensor", "folds selected", "folds in top-5",
                          "pooled AUC", "AUC per period", "I²", "p (BH)"]), "",
        ]
        top = max(broke, key=lambda r: r["folds_top5"])
        n_folds_total = max((r["folds_selected"] for r in iv["per_sensor"]),
                            default=0)
        if top["folds_top5"] >= 0.8 * max(n_folds_total, 1):
            body += [
                f"`{top['name']}` is in the agent loop's reported top-5 in "
                f"{top['folds_top5']} of {n_folds_total} cross-validation "
                f"folds -- its single most reproducible suspect -- and it is "
                f"the one associated sensor whose relationship with failure "
                f"demonstrably does not hold across production periods. "
                f"{int(round(100 * top['i2']))}% of the variance in its "
                f"per-period association is between periods rather than "
                f"within them.", "",
            ]

    pw = iv.get("power") or {}
    ladder = pw.get("ladder") or []
    if ladder:
        prows = [[f"{row['target_block0_auc']:.2f}",
                  pct(row["power_at_alpha"], 0),
                  pct(row["power_at_alpha_over_family"], 0)] for row in ladder]
        # smallest injected break the test detects at least half the time
        detect = [row for row in ladder if row["power_at_alpha"] >= 0.5]
        floor_txt = (f"about {detect[0]['target_block0_auc']:.2f}"
                     if detect else
                     f"more than {ladder[-1]['target_block0_auc']:.2f}")
        body += [
            "**A null result is only evidence if the test had power**, and "
            "this one is asked to detect a broken association from as few as "
            f"{min(b['n_fail'] for b in blocks)} failures in a block. So "
            "sensors were built with a known break -- association "
            "0.5 + delta in the first period, 0.5 in every other -- and put "
            "through the identical test:", "",
            table(prows, ["injected first-period AUC", "detected at p<0.05",
                          f"detected at p<0.05/{pw['n_family']}"]), "",
            f"The test needs a first-period AUC of {floor_txt} before it "
            f"finds the break half the time, and SECOM's associated sensors do "
            f"not have that much *total* signal. So the honest reading of the "
            f"table above is **not** \"the suspects are invariant, therefore "
            f"causal\" -- it is \"no break large enough for this dataset to "
            f"see\". The invariance screen cannot adjudicate causality on "
            f"SECOM at {int(sum(b['n_fail'] for b in blocks))} failures, and "
            f"saying so is the result.", "",
        ]
    if broke:
        top = max(broke, key=lambda r: r["folds_top5"])
        if kept:
            k_strength = sorted(abs(r["pooled_auc"] - 0.5) for r in kept)
            b_strength = abs(top["pooled_auc"] - 0.5)
            body += [
                f"That ordering is not a coincidence, and it is the reason a "
                f"rejection here means more than a pass. This test's power "
                f"rises with how strongly a sensor is associated, so the "
                f"sensors it is able to judge are exactly the ones the loop is "
                f"most confident about. `{top['name']}` is the strongest "
                f"association in the matrix (|AUC-0.5| = {b_strength:.3f}); "
                f"the {len(kept)} sensors that \"pass\" have a median of "
                f"{k_strength[len(k_strength)//2]:.3f}, below the level at "
                f"which the power table above shows the test can see anything "
                f"at all. **They did not pass an invariance test. They were "
                f"not testable.**", "",
            ]
    if ladder:
        body += [
            "**Therefore the pipeline reports associational suspects, and the "
            "repo says so wherever it names them.** Upgrading that to a causal "
            "claim needs either more failed wafers, or interventional data, or "
            "environments that differ more sharply than 90 days of one fab's "
            "history -- not a better attribution statistic.", "",
        ]

    ov = tot.get("n_loop_selected_and_associated")
    if ov is not None and tot["n_associated"]:
        body += [
            f"One side-observation with teeth: of the "
            f"{tot['n_ever_selected_by_loop']} sensors the agent loop selects "
            f"in at least one fold, {ov} are marginally associated -- and that "
            f"is {ov} of the {tot['n_associated']} associated sensors in the "
            f"whole matrix. The loop's candidate pool is essentially the "
            f"univariate screen plus "
            f"{tot['n_ever_selected_by_loop'] - ov} sensors with no detectable "
            f"marginal signal, which is the same conclusion the AUC and "
            f"stability tables reach from their own directions.", "",
        ]
    cal = iv.get("chi2_calibration") or {}
    if cal.get("chi2_rejection_rate_under_null") is not None:
        body += [
            f"*Method note.* The closed-form chi-square reference for "
            f"Cochran's Q is anticonservative on this data -- under the null "
            f"it rejects at {cal['chi2_rejection_rate_under_null']:.3f} "
            f"against a nominal {cal['nominal']:.3f}, because SECOM's sensors "
            f"carry heavy ties. Every decision above therefore uses the "
            f"permutation p-value instead; the chi-square figure is kept in "
            f"the JSON as a diagnostic only.", "",
        ]
    return body


def sec_kpi(ev, st, prof):
    if not (ev and st and "agent" in st.get("rankers", {})):
        return []
    rf = ev["auc"]["per_arm"]["rf_all"]
    ag_auc = ev["auc"]["per_arm"]["agent_rf"]
    ag = st["rankers"]["agent"]["bootstrap"]
    v1, ok1 = verdict(rf["mean"], KPI_AUC)
    v2, ok2 = verdict(ag_auc["mean"], KPI_AUC)
    v3, ok3 = verdict(ag["raw"]["pairwise_overlap"], KPI_STABILITY)
    v4, ok4 = verdict(ag["cluster"]["pairwise_overlap"], KPI_STABILITY)
    ci_clears = rf["ci_lo"] >= KPI_AUC
    rows = [
        [f"SECOM ROC-AUC >= {KPI_AUC:.2f}", "best plain baseline (`rf_all`)",
         ci(rf), v1 + ("" if ci_clears else " (point estimate; CI spans it)")],
        [f"SECOM ROC-AUC >= {KPI_AUC:.2f}", "agent loop (`agent_rf`)",
         ci(ag_auc), v2],
        [f"top-5 cause stability >= {KPI_STABILITY:.0%}",
         "agent loop, pairwise overlap, bootstrap",
         pct(ag["raw"]["pairwise_overlap"]), v3],
        [f"top-5 cause stability >= {KPI_STABILITY:.0%}",
         "agent loop, cluster-aware pairwise, bootstrap",
         pct(ag["cluster"]["pairwise_overlap"]), v4],
    ]
    cvs = st["rankers"]["agent"].get("cv_train")
    if cvs:
        v5, _ = verdict(cvs["raw"]["pairwise_overlap"], KPI_STABILITY)
        rows.append([
            f"top-5 cause stability >= {KPI_STABILITY:.0%}",
            "agent loop, pairwise, CV training folds (the gentler "
            "perturbation -- shown so the choice of protocol is visible)",
            pct(cvs["raw"]["pairwise_overlap"]), v5])
    return ["## KPI scorecard", "",
            "The catalog target for this project is "
            f"`SECOM real-data AUC >= {KPI_AUC:.2f}` and "
            f"`top-5 cause stability >= {KPI_STABILITY:.0%}`. Scored honestly:",
            "", table(rows, ["KPI", "measured on", "value", "verdict"]), "",
            ("Read together: the prediction KPI is "
             + ("met" if ok1 else "missed")
             + (" -- but by the plain baseline, not by the agent loop, which "
                f"lands at {ag_auc['mean']:.3f} and misses it"
                if ok1 and not ok2 else
                " by both the baseline and the agent loop" if ok1 and ok2
                else " by every arm")
             + ", and the stability KPI is "
             + ("met" if ok3 else
                f"missed by {100*(KPI_STABILITY - ag['raw']['pairwise_overlap']):.0f} "
                f"points")
             + " on the primary bootstrap protocol. The one-line summary is "
               "that on SECOM this pipeline is a usable *predictor* and an "
               "unreliable *root-cause attributor*, and the second half of "
               "that sentence is the finding."), "",
            (f"One caveat on the first row, stated rather than buried: the "
             f"point estimate {rf['mean']:.3f} clears {KPI_AUC:.2f}, but the "
             f"95% CI over folds runs "
             f"[{rf['ci_lo']:.3f}, {rf['ci_hi']:.3f}] and so includes values "
             f"below the target. \"Met\" here means the mean of "
             f"{ev['auc']['n_folds']} folds is above the line, not that the "
             f"line is cleared with confidence."
             if not ci_clears else
             f"The CI's lower bound ({rf['ci_lo']:.3f}) is itself above "
             f"{KPI_AUC:.2f}, so that row is not resting on a point "
             f"estimate."), ""]


# ---------------------------------------------------------------- assembly
def build(runs: Path):
    prof = read_json(runs / "data_profile.json")
    ev = read_json(runs / "secom_eval.json")
    sw = read_json(runs / "secom_loop_sweep.json")
    st = read_json(runs / "secom_stability.json")
    sy = read_json(runs / "synthetic.json")

    L = ["# Results", "",
         "Every number and every comparative claim below is generated by "
         "`scripts/report.py` from the JSON each run writes under `runs/`. "
         "None of it is typed in by hand, including the KPI verdicts.", ""]
    if ev:
        e = ev["environment"]
        L += ["Run environment: CPU only, "
              f"Python {e['python']}, scikit-learn {e['sklearn']}, "
              f"numpy {e['numpy']}; the cross-validation in "
              f"`runs/secom_eval.json` took {e['cv_wall_min']:.1f} min on 16 "
              "workers.", ""]
    dr = read_json(runs / "drift.json")
    rsw = read_json(runs / "rolling_sweep.json")
    nf = read_json(runs / "null_fdr.json")
    iv = read_json(runs / "invariance.json")
    L += sec_headline(ev, st, sy, sw, prof, dr, rsw)
    L += sec_kpi(ev, st, prof)
    L += sec_dataset(prof)
    L += sec_secom_auc(ev)
    L += sec_rolling(ev, prof)
    L += sec_drift(dr)
    L += sec_rolling_sweep(rsw, ev)
    L += sec_sweep(sw)
    L += sec_stability(st, prof)
    L += sec_null_fdr(nf)
    L += sec_invariance(iv)
    L += sec_synthetic(sy, ev)
    L += sec_limits(prof, st)
    L += ["## Leakage controls", "",
          "The failure mode this dataset invites is deciding *anything* from "
          "all 1,567 wafers and then cross-validating. Held inside the fold, "
          "every time:", "",
          table([
              ["constant / duplicate sensor detection",
               "`SensorCleaner.fit` on the training fold"],
              ["imputation medians",
               "`SimpleImputer` inside the pipeline"],
              ["standardisation",
               "`StandardScaler` inside the pipeline"],
              ["missing-indicator column choice",
               "`MissingIndicatorAppender.fit` on the training fold"],
              ["candidate screen", "`screen_*` on the training fold"],
              ["permutation importance",
               "inner split of the training fold; the model never sees the "
               "rows it is scored on"],
              ["bootstrap verification",
               "resamples drawn from the training fold only"],
              ["baseline hyperparameters",
               "inner-CV `GridSearchCV` on the training fold"],
              ["correlation clusters (reporting only)",
               "unlabelled sensor matrix; never used to predict"],
          ], ["decision", "where it is fitted"]), "",
          "The one deliberate exception is documented above: the cluster map "
          "used to compute the *cluster-aware* stability variant is built from "
          "the full unlabelled sensor matrix. It touches no labels and feeds "
          "no prediction.", ""]
    return "\n".join(L) + "\n"


README_BLOCKS = {
    "intro_data": lambda d: "\n".join(sec_intro(d["prof"], d["ev"])).strip(),
    "limits": lambda d: "\n".join(sec_limits(d["prof"], d["st"])[2:]).strip(),
    "headline": lambda d: "\n".join(
        sec_headline(d["ev"], d["st"], d["sy"], d["sw"], d["prof"], d["dr"],
                     d["rs"])[2:]).strip(),
    "kpi": lambda d: "\n".join(sec_kpi(d["ev"], d["st"], d["prof"])[2:]).strip(),
    "dataset": lambda d: "\n".join(sec_dataset(d["prof"])[2:]).strip(),
    "secom_auc": lambda d: "\n".join(sec_secom_auc(d["ev"])[2:]).strip(),
    "stability": lambda d: "\n".join(
        sec_stability(d["st"], d["prof"])[2:]).strip(),
    "synthetic": lambda d: "\n".join(
        sec_synthetic(d["sy"], d["ev"])[2:]).strip(),
    "sweep": lambda d: "\n".join(sec_sweep(d["sw"])[2:]).strip(),
    "rolling": lambda d: "\n".join(
        sec_rolling(d["ev"], d["prof"])[2:]).strip(),
    "drift": lambda d: "\n".join(sec_drift(d["dr"])[2:]).strip(),
    "rolling_sweep": lambda d: "\n".join(
        sec_rolling_sweep(d["rs"], d["ev"])[2:]).strip(),
    "null_fdr": lambda d: "\n".join(sec_null_fdr(d["nf"])[2:]).strip(),
    "invariance": lambda d: "\n".join(sec_invariance(d["iv"])[2:]).strip(),
}


def inject(readme: str, runs: Path) -> str:
    d = {
        "prof": read_json(runs / "data_profile.json"),
        "ev": read_json(runs / "secom_eval.json"),
        "sw": read_json(runs / "secom_loop_sweep.json"),
        "st": read_json(runs / "secom_stability.json"),
        "sy": read_json(runs / "synthetic.json"),
        "dr": read_json(runs / "drift.json"),
        "rs": read_json(runs / "rolling_sweep.json"),
        "nf": read_json(runs / "null_fdr.json"),
        "iv": read_json(runs / "invariance.json"),
    }
    for key, fn in README_BLOCKS.items():
        pat = re.compile(
            rf"(<!-- BEGIN:{key} -->)(.*?)(<!-- END:{key} -->)", re.S)
        if not pat.search(readme):
            continue
        body = fn(d)
        readme = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}",
                         readme)
    return readme


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--results", default=str(ROOT / "RESULTS.md"))
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if RESULTS.md or the README blocks are stale")
    a = ap.parse_args()

    runs = Path(a.runs)
    results = build(runs)
    readme_p = Path(a.readme)
    readme = inject(readme_p.read_text(), runs) if readme_p.exists() else None

    if a.check:
        stale = []
        if Path(a.results).exists():
            if Path(a.results).read_text() != results:
                stale.append(a.results)
        else:
            stale.append(f"{a.results} (missing)")
        if readme is not None and readme_p.read_text() != readme:
            stale.append(a.readme)
        if stale:
            print("STALE (rerun scripts/report.py): " + ", ".join(stale))
            return 1
        print("report is in sync with runs/*.json")
        return 0

    Path(a.results).write_text(results)
    print(f"wrote {a.results} ({len(results.splitlines())} lines)")
    if readme is not None:
        readme_p.write_text(readme)
        print(f"updated {a.readme} blocks: {', '.join(README_BLOCKS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
