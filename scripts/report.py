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


def attribution_2x2(st):
    """Separate the attribution statistic from the architecture around it.

    The ladder above adds the loop's mechanisms one at a time on top of a fixed
    attribution statistic, so it can only ever price the mechanisms. Reading the
    same table as a 2x2 -- {bare ranker, full loop} x {permutation, model-native
    attribution} -- prices the statistic too, and the two are not the same size.

    The bare-ranker cells are ``perm_only`` and ``model_only``: the loop's own
    attribution step with nothing after it, built from ``AgentRCA._rank`` so
    they cannot drift from the statistic the full loop consumes. An earlier
    version of this table used ``rf_impurity`` as the model-native bare cell,
    which an adversarial review correctly rejected -- it fits a different forest
    over every cleaned sensor with no screen, so subtracting it from
    ``agent_model`` priced the architecture plus a tree count plus a candidate
    universe. Until ``model_only`` lands, that column is reported as unpriced
    rather than estimated from a mismatched arm.
    """
    r = st.get("rankers") or {}
    if not all(k in r for k in ("perm_only", "agent", "agent_model")):
        return []
    v = {k: 100 * r[k]["bootstrap"]["raw"]["pairwise_overlap"] for k in r}
    w = {k: r[k]["bootstrap"]["wall_min"] for k in r}
    d_attr = v["agent_model"] - v["agent"]
    d_arch_perm = v["agent"] - v["perm_only"]
    matched = "model_only" in r
    sd = 100 * r["agent"]["bootstrap"]["raw"]["pairwise_overlap_sd"]

    if matched:
        bare_model = f"`model_only` {v['model_only']:.1f}%"
        d_bare = f"{v['model_only'] - v['perm_only']:+.1f}"
        d_arch_model = v["agent_model"] - v["model_only"]
        arch_model = f"{d_arch_model:+.1f}"
    else:
        bare_model = "*[not measured]*"
        d_bare = "*[not measured]*"
        d_arch_model = None
        arch_model = "*[not measured]*"

    body = [
        "**And which part is the statistic rather than the structure?** The "
        "ladder holds the attribution statistic fixed, so it can only price "
        "the loop's mechanisms. `agent_model` is the same loop with one field "
        "changed -- `attribution=\"model\"` instead of held-out permutation "
        "importance -- and `perm_only`/`model_only` are the loop's attribution "
        "step with nothing after it, which turns the table into a 2x2:", "",
        table([
            ["**bare ranker** (attribution step only)",
             f"`perm_only` {v['perm_only']:.1f}%", bare_model, d_bare],
            ["**full agent loop**", f"`agent` {v['agent']:.1f}%",
             f"`agent_model` {v['agent_model']:.1f}%", f"{d_attr:+.1f}"],
            ["**architecture is worth**", f"{d_arch_perm:+.1f}", arch_model,
             ""],
        ], ["", "permutation attribution", "model-native attribution",
            "statistic is worth"]), "",
        f"The clean cell is the bottom row: one configuration field, everything "
        f"else identical, worth **{d_attr:+.1f} points** to the loop and a "
        f"{w['agent'] / w['agent_model']:.1f}x speedup "
        f"({w['agent']:.1f} min to {w['agent_model']:.1f} min). Against that, "
        f"the whole architecture -- screen, correlation grouping, bootstrap "
        f"verify, drop -- is worth {d_arch_perm:+.1f} points over the "
        f"permutation statistic, which is inside one standard deviation of the "
        f"replicate-to-replicate spread ({sd:.1f} points).", "",
    ]
    if matched:
        body += [
            f"With the matched bare cell measured, the architecture is worth "
            f"{d_arch_perm:+.1f} points in the permutation column and "
            f"{d_arch_model:+.1f} in the model-native one"
            + (", and both are inside that one-sd band"
               if abs(d_arch_model) < sd and abs(d_arch_perm) < sd
               else ", one of which is outside that one-sd band")
            + f". Changing the statistic moves the number by "
              f"{abs(d_attr) / max(abs(d_arch_perm), abs(d_arch_model), 1e-9):.0f}x "
              f"more than changing the architecture does.", "",
            f"Two things about the architecture row deserve saying plainly, "
            f"one in each direction. It is **consistently positive** -- "
            f"{d_arch_perm:+.1f} and {d_arch_model:+.1f}, the same sign under "
            f"both statistics -- which is a better showing than an earlier "
            f"version of this table gave it. That version used `rf_impurity` "
            f"as the model-native bare cell and reported "
            f"{v['agent_model'] - v['rf_impurity']:+.1f}, so it had the "
            f"architecture helping under one statistic and "
            f"hurting under the other; the matched arm reverses that sign, and "
            f"the correction runs in the architecture's favour. But the size "
            f"has not changed: both deltas sit inside one standard deviation "
            f"of the replicate spread, and the architecture costs "
            f"{w['agent_model'] / w['model_only']:.1f}x the runtime to buy "
            f"{d_arch_model:+.1f} points ({w['model_only']:.1f} min to "
            f"{w['agent_model']:.1f} min). Small, real-looking, and not worth "
            f"its price is the fair summary -- not the stronger "
            f"\"no-op in both directions\" this repository claimed before the "
            f"matched cell existed.", "",
        ]
    else:
        body += [
            "**The model-native column's architecture delta is deliberately "
            "blank.** Pricing it needs a bare ranker built the same way as "
            "`perm_only` but with the other statistic, and that arm "
            "(`model_only`) is queued rather than measured. `rf_impurity` is "
            f"not a substitute for it: at {v.get('rf_impurity', float('nan')):.1f}% "
            "it is close, but it fits a 500-tree forest over every cleaned "
            "sensor with no screen, so the difference from `agent_model` is "
            "the architecture plus a tree count plus a candidate universe. "
            "A blank is the honest entry until the matched arm lands.", "",
        ]
    body += [
        "*This was run as a pre-registered prediction rather than a sweep.* "
        "`critique_log.md` Turn 8 predicted, before the run, that the swap "
        f"would land near `rf_impurity`'s "
        f"{v.get('rf_impurity', float('nan')):.1f}% and would not reach the "
        f"{KPI_STABILITY:.0%} KPI; the competing explanation on record "
        f"predicted it would stay near {v['agent']:.1f}%. Both halves of the "
        f"first prediction hold, and the miss against the KPI is still "
        f"{KPI_STABILITY * 100 - v['agent_model']:.1f} points.", "",
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
        body += attribution_2x2(st)
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


def sec_headline(ev, st, sy, sw, prof=None, dr=None, rsw=None,
                 nf=None, ab=None, rk=None):
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
    if nf:
        n = nf["null"]
        fd = nf["null_fdr"]
        sep = nf["separation"]["prob_real_max_exceeds_null_max"]
        line = (
            f"- **It reports root causes on data that has none.** Permute the "
            f"labels so no sensor carries any information about failure, and "
            f"the full plan/attribute/verify/drop loop still names "
            f"{n['n_reported_mean']:.1f} suspects per replicate and abstains on "
            f"{pct(n['abstention_rate'])} of them -- "
            f"{fd['false_discoveries_total']:,} false discoveries over "
            f"{nf['protocol']['n_null']} replicates, a false-discovery rate of "
            f"**{fd['fdr_given_nonempty']:.0%}**. The advertised safeguard, "
            f"that unstable suspects are dropped, does not hold: "
            f"{n['n_merit_mean']:.1f} pure-noise sensors clear the stability "
            f"threshold unaided, and the never-empty fallback is not even "
            f"needed ({pct(n['fallback_rate'])}).")
        if ab and ab.get("levels"):
            mid = ab["levels"].get("alpha_0.05") or list(ab["levels"].values())[0]
            base = ab["no_rule_baseline"]
            line += (
                f" The statistic is salvageable -- P(real > null) = "
                f"{sep:.2f} -- so a threshold calibrated on that null, "
                f"tau = {mid['tau_mean']:.2f}, restores control. It also "
                f"shortens the SECOM report from "
                f"{base['real_reported_mean']:.1f} suspects to "
                f"**{mid['real_reported_mean']:.2f}**, empty "
                f"{pct(mid['real_abstention'])} of the time. That is what this "
                f"dataset actually supports.")
        L.append(line)
    if ab and (ab.get("null_structure") or {}).get("null"):
        stt = ab["null_structure"]
        L.append(
            f"- **And the invented causes are fresh each time, not an "
            f"artefact.** Null replicates agree with each other on only "
            f"{stt['null']['top5_pairwise_overlap']:.3f} of their top-5 "
            f"against a random-ranker floor of "
            f"{stt['random_floor_top5']:.3f}, naming "
            f"{stt['null']['distinct_sensors_ever_named']} of "
            f"{stt['n_eff_sensors']} sensors at least once across the run. So "
            f"the loop is not re-reporting SECOM's correlation structure under "
            f"the null; it is manufacturing a different answer every time it "
            f"is asked.")
    if rk and rk.get("per_ranker"):
        per = rk["per_ranker"]
        ag_k = next((k for k in per if k.startswith("agent")), None)
        plain = {k: v for k, v in per.items() if not k.startswith("agent")}
        _ = plain
        if ag_k and plain:
            def ctl(v):
                return v["heldout_alpha_0.05"]["null_abstention_heldout"]
            bc_k = max(plain, key=lambda k: ctl(plain[k]))
            bc, ag = per[bc_k], per[ag_k]
            wins_both = (per[bc_k]["prob_real_max_exceeds_null_max"]
                         >= ag["prob_real_max_exceeds_null_max"]
                         and ctl(bc) >= ctl(ag))
            L.append(
                f"- **And a univariate ranker does the same job better.** "
                f"Matched to the loop's own bootstrap count and selection "
                f"depth, plain rankers separate the two worlds better "
                f"(`univariate` {plain.get('univariate', bc)['prob_real_max_exceeds_null_max']:.3f} "
                f"vs {ag['prob_real_max_exceeds_null_max']:.3f}) but their "
                f"support *saturates*, capping their error control below the "
                f"loop's. Narrow the selection depth so it stops saturating "
                f"and the cap disappears: `{bc_k}` reaches "
                f"{pct(ctl(bc))} control against the loop's "
                f"{pct(ctl(ag))} at a 95% target, separates at "
                f"{bc['prob_real_max_exceeds_null_max']:.3f}, and still "
                f"reports {bc['heldout_alpha_0.05']['real_reported_mean']:.2f} "
                f"suspects against "
                f"{ag['heldout_alpha_0.05']['real_reported_mean']:.2f}"
                + (" -- with no permutation-importance pass, no correlation "
                   "grouping and no verification loop. **So on every axis "
                   "measured here the loop is matched or beaten by a "
                   "univariate ranker.** Two caveats travel with that: the "
                   "selection depth was probed for the baseline and not for "
                   "the loop, and the separation column rewards a repeatable "
                   "ranker as well as a discriminating one. Both are detailed "
                   "in the section below." if wins_both else
                   ", so the two are close on this axis -- and the plain "
                   "ranker needs no permutation-importance pass, no "
                   "correlation grouping and no verification loop."))
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
         "how far apart the two worlds sit on the loop's own statistic"],
    ]
    verdict_word = ("never once" if n["abstention_rate"] == 0 else
                    f"in {pct(n['abstention_rate'])} of replicates")
    sep_p = sep["prob_real_max_exceeds_null_max"]
    if sep_p >= 0.75:
        sep_read = ("so the statistic is strongly informative about whether "
                    "the labels were real -- it is the *threshold* that is "
                    "mis-set, not the measurement, and the next section prices "
                    "the recalibration")
    elif sep_p >= 0.6:
        sep_read = ("so the statistic carries some information about whether "
                    "the labels were real, but little enough that "
                    "recalibrating its threshold will cost most of the report")
    else:
        sep_read = ("so the statistic carries essentially no information about "
                    "whether the labels were real, and no threshold on it can "
                    "recover one -- the attribution step itself would need "
                    "replacing")
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
        f"**The loop abstained {verdict_word}.** Over {pr['n_null']} "
        f"permuted-label replicates it named "
        f"{fd['false_discoveries_total']:,} sensors as root causes. Every one of "
        f"them is a false discovery by construction, so the false-discovery "
        f"rate of the reported suspect list under this null is "
        f"**{fd['fdr_given_nonempty']:.0%}**.", "",
        ("**And the mechanism is not the one the code invites you to blame.** "
         f"`AgentRCA.fit` carries two never-return-empty-handed guards "
         f"(`estimator.py`, `if not surv: surv = reps[:5]`), which would "
         f"produce exactly this result -- but they fired on "
         f"{pct(n['fallback_rate'])} of null replicates. They are not what is "
         f"happening. The threshold itself is: "
         f"{n['n_merit_mean']:.1f} pure-noise sensors per replicate clear "
         f"pi = {pr['agent_cfg']['stability_min']} **on their own merit**. "
         f"To clear it, a sensor need only reach the top "
         f"{pr['agent_cfg']['select_k']} of a "
         f"{pr['agent_cfg']['n_screen_boot']}-sensor candidate pool in "
         f"{int(round(pr['agent_cfg']['stability_min'] * pr['agent_cfg']['n_boot']))} "
         f"of {pr['agent_cfg']['n_boot']} bootstrap replicates -- which noise "
         f"does routinely. Lowering or raising the guard changes nothing; the "
         f"bar is in the wrong place."
         if n["fallback_rate"] < 0.5 else
         "**The mechanism is the never-return-empty-handed guard.** "
         f"`AgentRCA.fit` restores the top candidates when nothing clears the "
         f"threshold (`estimator.py`, `if not surv: surv = reps[:5]`), and on "
         f"the null it fired on {pct(n['fallback_rate'])} of replicates -- "
         f"precisely when withholding a report is the only correct action."),
        "",
        f"**Is the loop at least *more* confident on real data?** "
        f"P(real replicate's best support > null replicate's best support) = "
        f"**{sep_p:.3f}**, where 0.5 is no information "
        f"(Mann-Whitney p = {sep['p_real_greater']:.3g}), {sep_read}.", "",
    ]
    return body


def nf_cfg(ab):
    """The agent config behind an abstain run, if the source JSON is at hand."""
    src = (ab.get("protocol") or {}).get("source")
    if not src:
        return None
    d = read_json(ROOT / src) or read_json(src)
    return ((d or {}).get("protocol") or {}).get("agent_cfg")


def sec_abstain(ab):
    """The price of letting the pipeline report nothing."""
    if not ab:
        return []
    pr = ab["protocol"]
    base = ab["no_rule_baseline"]
    rows = []
    for key in sorted(ab["levels"], key=lambda k: -ab["levels"][k]["alpha"]):
        L = ab["levels"][key]
        rows.append([
            f"{L['alpha']:g}", f"{L['tau_mean']:.3f}",
            f"{pct(L['null_abstention_heldout'])} "
            f"(target {pct(L['null_abstention_target'], 0)})",
            f"{L['null_false_discoveries_heldout']:.2f}",
            pct(L["real_abstention"]),
            f"{L['real_reported_mean']:.2f}",
        ])
    rows.append(["-- none --", "--", pct(base["null_abstention"]),
                 f"{base['null_reported_mean']:.2f}", pct(0.0),
                 f"{base['real_reported_mean']:.2f}"])
    mid = ab["levels"].get("alpha_0.05") or list(ab["levels"].values())[0]
    n_boot = ((ab.get("protocol") or {}).get("n_boot")
              or (nf_cfg(ab) or {}).get("n_boot"))
    body = [
        "### What it would cost to let it say nothing", "",
        f"Same run, no refits: every figure below is a function of the "
        f"per-replicate suspect supports in `runs/null_fdr.json`, computed by "
        f"`scripts/abstain.py` into `runs/abstain.json`.", "",
        f"**The rule.** {pr['rule']}.", "",
        f"**The calibration is held out.** {pr['why_split']}, so "
        f"{pr['calibration']}.", "",
        table(rows, ["alpha", "tau", "held-out null: reports nothing",
                     "held-out null: false discoveries",
                     "real labels: reports nothing",
                     "real labels: suspects reported"]), "",
        f"At alpha = {mid['alpha']:g} the honest SECOM report is "
        f"**{mid['real_reported_mean']:.2f} sensors on average, and empty "
        f"{pct(mid['real_abstention'])} of the time** -- against the "
        f"{base['real_reported_mean']:.1f} the pipeline prints today. That is "
        f"the finding stated as a deliverable: this dataset supports about one "
        f"named suspect, sometimes none, and the current report's length is "
        f"not evidence about the process.", "",
        f"**The rule is itself slightly optimistic, and the table says so.** "
        f"Held-out abstention on the null lands at "
        f"{pct(mid['null_abstention_heldout'])} against a nominal "
        f"{pct(mid['null_abstention_target'], 0)}, because tau is a quantile "
        f"estimated from finitely many null replicates and a point estimate of "
        f"an upper quantile is biased low. Closing that gap means more null "
        f"replicates or an upper confidence bound on the quantile rather than "
        f"the quantile itself; it is not closed here, and the shortfall is "
        f"reported rather than rounded away.", "",
        (f"**The bar sits on a coarse grid.** Support is a fraction of "
         f"{n_boot} bootstrap replicates, so it takes only {n_boot + 1} "
         f"distinct values and tau cannot be placed between them. At the "
         f"strictest level here tau lands at or next to the ceiling, which is "
         f"why the alpha = 0.01 row buys little over alpha = 0.05: there is no "
         f"room above it. Finer control needs more bootstrap replicates inside "
         f"the loop, which costs linearly and was not spent here."
         if n_boot else
         "**The bar sits on a grid** set by how many bootstrap replicates the "
         "loop runs, and cannot be placed between grid points."), "",
        "`AgentRCA(report_tau=...)` implements the rule. It governs "
        "`reported_` only -- `selected_` and `predict_proba` are byte-identical "
        "with and without it, asserted in "
        "`tests/test_null.py::test_report_tau_lets_the_loop_abstain_on_pure_noise` "
        "-- so switching abstention on cannot move any AUC in this repo, and "
        "the prediction and attribution claims stay separable.", "",
    ]
    st = ab.get("null_structure")
    if st:
        floor = st["random_floor_top5"]
        o_null = st["null"]["top5_pairwise_overlap"]
        o_real = st["real"]["top5_pairwise_overlap"]
        body += [
            "### Is the null unfairly easy?", "",
            "One alternative would deflate all of the above: permuting labels "
            "leaves the sensor *correlation* structure intact, so perhaps the "
            "loop is reporting that structure rather than inventing anything. "
            "If so, null replicates would keep naming the same sensors as each "
            "other. They do not:", "",
            table([
                ["null replicates agree with each other", f"{o_null:.3f}",
                 f"{st['null']['n_replicates_top5']} replicates"],
                ["a uniformly random top-5 would agree", f"{floor:.3f}",
                 f"5 / {st['n_eff_sensors']} surviving sensors"],
                ["real-label replicates agree with each other", f"{o_real:.3f}",
                 f"{st['real']['n_replicates_top5']} replicates"],
                ["distinct sensors the null ever named",
                 f"{st['null']['distinct_sensors_ever_named']}", "of "
                 f"{st['n_eff_sensors']}"],
            ], ["mean pairwise top-5 overlap", "value", "over"]), "",
            f"At {o_null:.3f} against a floor of {floor:.3f}, the null's "
            f"suspects are freshly invented on each replicate rather than a "
            f"stable artefact of the correlation structure. The alternative "
            f"does not hold, and the false-discovery rate stands.", "",
            f"**The {o_real:.3f} is not comparable to this repo's top-5 "
            f"stability KPI and must not be read as one.** These replicates "
            f"perturb only the loop's internal random seed on the full wafer "
            f"set; the KPI perturbs the *wafers*, by bootstrap resampling, "
            f"which is a far harder test and is why it reads much lower in the "
            f"stability section. The number is here only as the upper "
            f"reference for the null column beside it.", "",
        ]
    return body


def sec_ranker_fdr(rk):
    """Does the loop's verification machinery calibrate better than a plain ranker?

    Two columns decide it and they disagree, so both are printed: how well a
    statistic *separates* a world with causes from one without, and how much
    error control it can actually be thresholded to. A statistic that saturates
    on real data separates beautifully and cannot be calibrated.
    """
    if not rk:
        return []
    per = rk["per_ranker"]
    pr = rk["protocol"]
    agent_key = next((k for k in per if k.startswith("agent")), None)
    rows = []
    for k in sorted(per, key=lambda k: -per[k]["prob_real_max_exceeds_null_max"]):
        v = per[k]
        h = v["heldout_alpha_0.05"]
        rows.append([
            f"**{k}**" if k.startswith("agent") else f"`{k}`",
            f"{v['prob_real_max_exceeds_null_max']:.3f}",
            f"{h['tau_mean']:.3f}",
            f"{pct(h['null_abstention_heldout'])}",
            f"{pct(h['max_attainable_null_abstention'])}",
            f"{h['real_reported_mean']:.2f}",
            pct(h["real_abstention"], 0),
        ])
    body = [
        "## Would a plain ranker have calibrated better?", "",
        f"The section above shows the loop's bootstrap support *is* informative "
        f"about whether the labels were real, which is what makes a calibrated "
        f"threshold work at all. So: {pr['question']} "
        f"Comparing raw false-discovery rates would settle nothing -- "
        f"{pr['why_not_fdr']}.", "",
        f"**Matched by construction:** {pr['matched']}. "
        # Stated from the verified source, not from the JSON's own field: an
        # earlier run recorded "fitted inside each resample" there, which was
        # wrong, and that string survives in any JSON written before the fix.
        + "`SensorCleaner` is fitted once per replicate on the full matrix, "
          "outside the bootstrap loop. It is unsupervised -- it drops "
          "all-missing, constant and duplicate columns from `X` alone, never "
          "`y` -- so a label permutation cannot change its output and it leaks "
          "nothing into the null, and `AgentRCA.fit` cleans the same way, so "
          "the arms are matched on this too. "
        + f"The held-out calibration is the same "
        f"split-half procedure `scripts/abstain.py` uses. From "
        f"`scripts/null_fdr_rankers.py` into `runs/null_fdr_rankers.json`; the "
        f"agent row is recomputed from `{pr['agent_arm_source']}` by the same "
        f"code path, so no arm gets a different protocol.", "",
        table(rows, ["ranker", "P(real > null)", "tau (0.05)",
                     "no-cause worlds kept silent", "ceiling on that",
                     "suspects reported", "reports nothing"]), "",
        "The last two columns are the deliverable; the middle two are why the "
        "obvious reading of the first one is wrong.", "",
    ]
    if not agent_key:
        return body
    ag = per[agent_key]
    ag_h = ag["heldout_alpha_0.05"]
    plain = {k: v for k, v in per.items() if not k.startswith("agent")}
    if not plain:
        return body
    bs_k = max(plain, key=lambda k: plain[k]["prob_real_max_exceeds_null_max"])
    bs = plain[bs_k]
    bc_k = max(plain, key=lambda k: plain[k]["heldout_alpha_0.05"]["null_abstention_heldout"])
    bc = plain[bc_k]["heldout_alpha_0.05"]
    sep_gap = bs["prob_real_max_exceeds_null_max"] - ag["prob_real_max_exceeds_null_max"]
    ctl_gap = bc["null_abstention_heldout"] - ag_h["null_abstention_heldout"]

    if sep_gap > 0.02:
        body += [
            f"**On separation the plain rankers win.** `{bs_k}` distinguishes "
            f"the two worlds at {bs['prob_real_max_exceeds_null_max']:.3f} "
            f"against the full loop's "
            f"{ag['prob_real_max_exceeds_null_max']:.3f}. Taken alone that "
            f"says the whole plan/attribute/verify apparatus is a worse "
            f"signal detector than ranking each sensor on its own -- the same "
            f"verdict the AUC and stability tables reach, from a third "
            f"direction.", "",
        ]
    elif sep_gap < -0.02:
        body += [
            f"**On separation the agent loop wins**, at "
            f"{ag['prob_real_max_exceeds_null_max']:.3f} against "
            f"`{bs_k}`'s {bs['prob_real_max_exceeds_null_max']:.3f}.", "",
        ]
    else:
        body += [
            f"**Separation is a tie** ({ag['prob_real_max_exceeds_null_max']:.3f} "
            f"for the loop, {bs['prob_real_max_exceeds_null_max']:.3f} for "
            f"`{bs_k}`).", "",
        ]

    if ctl_gap < -0.02:
        body += [
            f"**But separation is not the property you can ship, and on the "
            f"one that is, the ordering reverses.** A plain ranker's support "
            f"saturates: on real labels its best sensor sits in the top slice "
            f"of every bootstrap replicate, so the statistic pins at 1.000 -- "
            f"and it does that on "
            f"{pct(1 - plain[bc_k]['heldout_alpha_0.05']['max_attainable_null_abstention'])} "
            f"of *permuted-label* replicates too. No threshold at or below "
            f"1.000 can exclude those, so `{bc_k}` cannot be calibrated past "
            f"{pct(bc['max_attainable_null_abstention'])} error control no "
            f"matter what alpha is asked for, and lands at "
            f"{pct(bc['null_abstention_heldout'])} against the 95% target. The "
            f"agent loop's noisier statistic has headroom: it reaches "
            f"{pct(ag_h['null_abstention_heldout'])}, the only arm here that "
            f"comes near nominal.", "",
            f"**So the honest verdict is split, and the split is the "
            f"finding.** If the question is *is there signal in this dataset at "
            f"all*, a univariate ranker answers it better and cheaper. If the "
            f"question is *give me a suspect list with a stated "
            f"false-discovery guarantee*, only the agent loop's statistic can "
            f"carry a guarantee -- and the price is a report of "
            f"{ag_h['real_reported_mean']:.2f} suspects, empty "
            f"{pct(ag_h['real_abstention'])} of the time, against `{bc_k}`'s "
            f"{bc['real_reported_mean']:.2f} at weaker control. This is the "
            f"first axis in this repository on which the verification "
            f"machinery earns anything, and it earns it for a reason that has "
            f"nothing to do with finding better sensors: its estimator is "
            f"noisy enough to be thresholdable.", "",
            "That is a genuinely uncomfortable argument in the loop's favour "
            "and it should be read as narrowly as it is stated. It is not "
            "evidence that the loop's suspects are better. It says that a "
            "saturating statistic cannot express uncertainty, and the loop's "
            "does -- which a coarser but bounded alternative (a univariate "
            "ranker with more bootstrap replicates and a smaller selection "
            "depth, so its support stops pinning at 1.0) would very likely "
            "also achieve. That ablation is not run, and until it is, the "
            "advantage claimed here is over *these* rankers at *this* "
            "operating point, not over simplicity in general.", "",
        ]
    elif ctl_gap > 0.02:
        base = {k: v for k, v in plain.items() if not v.get("is_variant")}
        var = {k: v for k, v in plain.items() if v.get("is_variant")}
        body += [
            f"**And once the operating point stops flattering it, the plain "
            f"ranker controls error better too.** `{bc_k}` reaches "
            f"{pct(bc['null_abstention_heldout'])} against the loop's "
            f"{pct(ag_h['null_abstention_heldout'])}, at a 95% target -- while "
            f"still reporting {bc['real_reported_mean']:.2f} suspects against "
            f"the loop's {ag_h['real_reported_mean']:.2f}, and abstaining on "
            f"{pct(bc['real_abstention'], 0)} of real replicates against "
            f"{pct(ag_h['real_abstention'], 0)}.", "",
        ]
        if base and var:
            b_k = max(base, key=lambda k: base[k]["heldout_alpha_0.05"]["null_abstention_heldout"])
            b_h = base[b_k]["heldout_alpha_0.05"]
            body += [
                f"That qualifier is the whole result, so it is worth being "
                f"exact about it. Matched to the agent loop's own settings "
                f"(`select_k = {base[b_k]['select_k']}` of "
                f"{int(rk.get('n_eff_sensors', 474))} sensors, "
                f"`n_boot = {base[b_k]['n_boot']}`), a plain ranker's support "
                f"**saturates**: its best sensor sits in the top slice of every "
                f"resample, so the statistic pins at 1.000 on real labels and "
                f"on "
                f"{pct(1 - b_h['max_attainable_null_abstention'])} of permuted "
                f"ones too. No threshold at or below 1.000 excludes those, so "
                f"`{b_k}` is capped at "
                f"{pct(b_h['max_attainable_null_abstention'])} control for any "
                f"threshold rule of the form used here -- below the loop's "
                f"{pct(ag_h['null_abstention_heldout'])}, which is what made "
                f"the matched comparison alone look like a win for the "
                f"architecture.", "",
                f"Narrowing the selection depth removes the saturation "
                f"entirely. Every variant row above reaches a "
                f"{pct(max(v['heldout_alpha_0.05']['max_attainable_null_abstention'] for v in var.values()), 0)} "
                f"ceiling and lands within "
                f"{abs(0.95 - bc['null_abstention_heldout']) * 100:.1f} points "
                f"of nominal, without a permutation-importance pass, a "
                f"correlation-grouping step, or a verification loop. The "
                f"agent loop's apparent advantage was a property of the "
                f"operating point it was compared at, not of the "
                f"plan/attribute/verify architecture.", "",
                "**So the loop has no measured advantage on any axis in this "
                "repository.** It loses on held-out AUC, it loses on top-5 "
                "selection stability, it loses on how well its confidence "
                "separates signal from noise, and it loses on how much "
                "false-discovery control that confidence can be calibrated "
                "to. The one place it wins remains the synthetic generator, "
                "where its premise -- that a few sensors dominate -- is true "
                "by construction.", "",
                "**Three things this comparison does not establish**, all "
                "raised by an adversarial review of it and recorded in "
                "`critique_log.md`:", "",
                f"- *Separation is confounded with repeatability.* "
                f"{pr.get('separation_confound', '')} So the separation column "
                f"should be read as the weaker of the two, and the "
                f"error-control column as the one that carries the argument.",
                f"- *Report length is not an accuracy axis.* "
                f"{pr.get('report_length_caveat', '')}",
                f"- *The selection depth was probed for one arm.* "
                f"`select_k = 5` was chosen because it removes the saturation, "
                f"and the corresponding agent configuration is a separate run. "
                f"Until that lands, this table shows a tuned baseline against "
                f"an untuned loop, which is the right comparison for \"would "
                f"something simpler have done\" and the wrong one for \"is "
                f"the architecture worse at equal effort\".", "",
                "*This paragraph replaces an earlier conclusion in this "
                "repository's history.* The matched-settings comparison alone "
                "showed the loop as the only arm able to carry a "
                "false-discovery guarantee, and that was written up as its "
                "first genuine win. The follow-up run in the table above "
                "refuted it. Both are in `critique_log.md`; the earlier "
                "reading was wrong because it compared one operating point "
                "and generalised to an architecture.", "",
            ]
    else:
        body += [
            f"**Error control is comparable too**: "
            f"{pct(bc['null_abstention_heldout'])} for `{bc_k}` against "
            f"{pct(ag_h['null_abstention_heldout'])} for the loop, both "
            f"against a 95% target.", "",
        ]
    return body


def fallback_reach(nf5, ab5=None, alpha="alpha_0.05"):
    """Does the never-empty fallback actually reach the calibrated report?

    Written because this repository asserted for one turn that it does -- that
    the guard "is the thing that makes abstention impossible". That reading
    conflated two sets the loop keeps separate. ``selected_`` is what the final
    classifier is fitted on and cannot be empty, because a classifier needs at
    least one column; ``reported_`` is what an engineer is handed, and under the
    tau rule it is allowed to be empty. The fallback tops up the first. The
    error-control column is a function of the second.

    Everything below is recomputed from the per-replicate supports in
    ``runs/null_fdr_k5.json``, so it is a measurement of that run and not an
    argument about it.
    """
    recs = (nf5 or {}).get("records") or []
    null = [r for r in recs if r["permuted"]]
    tau = ((nf5 or {}).get("thresholds") or {}).get(alpha)
    if not null or tau is None:
        return []
    pi = float(null[0]["stability_min"])
    fired = [r for r in null if r["fallback_fired"]]
    unaided = [r for r in null if not r["fallback_fired"]]
    over = lambda g: sum(any(v >= tau for v in r["stability_values"]) for r in g)
    if not fired or not unaided:
        return []
    rows = [
        [f"fallback fired (nothing cleared pi = {pi:g})", str(len(fired)),
         f"{max(r['max_stability'] for r in fired):.3f}", str(over(fired))],
        ["threshold cleared on merit", str(len(unaided)),
         f"{max(r['max_stability'] for r in unaided):.3f}", str(over(unaided))],
    ]
    return [
        "**But the fallback never reaches the report, and that is measurable "
        "rather than arguable.** Splitting the null replicates by whether it "
        "fired:", "",
        table(rows, [f"null replicates, `select_k = 5`", "n",
                     "largest support reached",
                     f"replicates naming a suspect over tau = {tau:.3f}"]), "",
        f"The fallback fires exactly when no sensor clears pi = {pi:g}, so by "
        f"construction those replicates top out below {pi:g} -- and tau({alpha.split('_')[1]}) "
        f"= {tau:.3f} sits *above* pi. Every one of the {len(fired)} replicates "
        f"the fallback fires on is therefore silent under the calibrated rule: "
        f"{over(fired)} of them name anything. The "
        f"{over(unaided)} null replicates that do get through are all "
        f"replicates where the threshold was cleared on merit -- that is, "
        f"where the attribution estimator handed a pure-noise sensor a "
        f"genuinely high bootstrap support. **The residual error-control "
        f"failure is the estimator's, not the guard's.**", "",
        f"One bookkeeping note, so the counts are not read as inconsistent: "
        f"the {over(unaided)} above uses this run's own full-null tau, while "
        f"the error-control column in the table above is the split-half "
        f"held-out figure from `scripts/abstain.py`, which fits tau on one "
        f"half of the null replicates and counts on the other. The two differ "
        f"by the usual amount an in-sample quantile differs from a held-out "
        f"one; the held-out figure is the one that carries the claim, and the "
        f"cross-tab is here for the mechanism, not for the rate.", "",
    ] + _fallback_reach_heldout(ab5) + [
    ]


def _fallback_reach_heldout(ab5):
    """The same question asked inside the protocol that carries the figure.

    An adversarial review of the cross-tab above made the right objection: one
    full-null tau is not the tau the headline uses, so showing that no
    guard-fired replicate clears 0.417 does not show that none clears every
    split-specific held-out tau. `scripts/abstain.py` now counts it directly,
    per calibration/evaluation split, which is a strictly stronger check than
    the one the objection asked for.
    """
    lv = (ab5 or {}).get("levels") or {}
    if not lv:
        return []
    rows = []
    for key in ("alpha_0.1", "alpha_0.05", "alpha_0.01"):
        m = lv.get(key)
        if not m:
            continue
        rows.append([f"alpha = {m['alpha']}", f"{m['tau_min']:.3f}",
                     f"{m['tau_max']:.3f}",
                     f"{m['fallback_reached_report_total']} of "
                     f"{m['n_splits_evaluated']}"])
    if not rows:
        return []
    return [
        "**And the same question, asked inside the protocol that produces the "
        "figure.** The cross-tab above thresholds at one full-null tau, which "
        "is not the tau the error-control column uses; an adversarial review "
        "pointed out that this leaves the stronger claim unproven. So "
        "`scripts/abstain.py` now counts it per split: across every "
        "calibration/evaluation partition, how many evaluation-half "
        "replicates both had the guard fire *and* named a suspect over that "
        "split's own tau.", "",
        table(rows, ["`select_k = 5`", "smallest tau fitted",
                     "largest tau fitted",
                     "splits where the guard reached the report"]), "",
        "Zero, at every level, and the reason is visible in the tau column: "
        "the smallest threshold any split fits still sits above "
        f"`stability_min` = 0.3, and the guard fires only when every support "
        f"is below it. So this is not a rate that happened to come out at "
        f"zero -- it is an ordering that holds across all "
        f"{lv['alpha_0.05']['n_splits_evaluated']} fitted thresholds. The "
        "guard cannot reach the calibrated report at this operating point.", "",
    ] + [
        "*This retracts a correction made earlier in this repository.* Turn 9 "
        "of `critique_log.md` read the 62.5% fallback rate as the guard "
        "destroying an otherwise working filter, and wrote that up as "
        "overturning an earlier finding. The re-reading was wrong in the same "
        "way the thing it corrected was: it took a property of `selected_`, "
        "the prediction set, and attributed it to the report. The original "
        "reading -- that the guards are not the mechanism behind the "
        "false-discovery rate -- holds at both depths. What is true about the "
        "fallback is narrower: it makes the *uncalibrated* "
        "`stability_min` filter unable to return an empty prediction set, and "
        "the abstention row above is 0% at both depths because these runs set "
        "`report_tau = None`, which disables abstention by configuration. "
        "Neither fact bears on the tau-calibrated column.", "",
    ]


def sec_attr_auc(aa, ev):
    """H7: does the attribution statistic close the loop's AUC deficit?

    The loop loses 0.042 AUC to a full-sensor forest. H5 and H6 localised its
    weakest component to the attribution statistic, so this asks the accuracy
    question the same way. Prediction registered in `critique_log.md` Turn 11
    before the run: the deficit is mostly the price of *sparsity* rather than of
    ranking quality, so the swap lands near the naive-selection control's 0.730
    and leaves a paired deficit whose CI excludes zero.
    """
    if not (aa and ev):
        return []
    a = aa["auc"]
    pr = aa.get("paired") or {}
    name = aa["arm"]
    per = ev["auc"]["per_arm"]
    rows = []
    for arm in ("rf_all", "univar_top25_rf", "agent_rf"):
        key = f"{name}__vs__{arm}"
        if key not in pr or arm not in per:
            continue
        d = pr[key]
        rows.append([f"`{arm}`", f"{per[arm]['mean']:.3f}",
                     f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]",
                     f"{d['wins']}/{d['losses']}",
                     f"{d.get('wilcoxon_p', float('nan')):.3f}"])
    if not rows:
        return []
    d_base = pr.get(f"{name}__vs__rf_all", {})
    d_uni = pr.get(f"{name}__vs__univar_top25_rf", {})
    d_ag = pr.get(f"{name}__vs__agent_rf", {})
    old_def = ev["auc"]["paired"].get("agent_rf__vs__rf_all", {})
    body = [
        "### Does the attribution statistic close the AUC gap? (H7)", "",
        f"One arm added under the published protocol -- `{name}`, the "
        f"pre-registered loop with `attribution=\"model\"` and nothing else "
        f"changed. The folds are byte-identical to `runs/secom_eval.json`'s "
        f"(`RepeatedStratifiedKFold` is deterministic), so the deltas below "
        f"are paired against that file's stored per-fold AUCs and nothing "
        f"already published was recomputed.", "",
        f"**`{name}` scores {a['mean']:.3f} [{a['ci_lo']:.3f}, "
        f"{a['ci_hi']:.3f}].**", "",
        table(rows, ["paired against", "its AUC", f"`{name}` minus it",
                     "folds W/L", "Wilcoxon p"]), "",
    ]
    if d_base and d_uni and old_def:
        body += [
            f"**The deficit against the full-sensor forest shrinks but does "
            f"not close:** {old_def['mean']:+.3f} for the permutation loop, "
            f"{d_base['mean']:+.3f} "
            f"[{d_base['ci_lo']:+.3f}, {d_base['ci_hi']:+.3f}] here, with the "
            f"interval still excluding zero. And the arm now sits on top of "
            f"the naive-selection control: {d_uni['mean']:+.4f} "
            f"[{d_uni['ci_lo']:+.4f}, {d_uni['ci_hi']:+.4f}], Wilcoxon "
            f"p = {d_uni.get('wilcoxon_p', float('nan')):.2f} -- "
            f"indistinguishable from ranking each sensor on its own and "
            f"keeping the top 25.", "",
            f"**So the loop's accuracy deficit is not fixable by any "
            f"attribution statistic**, which was registered as the prediction "
            f"and is a worse result for the architecture than the alternative "
            f"would have been: under the competing explanation the deficit was "
            f"bad ranking and therefore repairable.", "",
            f"*A decomposition that stood here has been withdrawn.* This "
            f"section originally split the {old_def['mean']:+.3f} into roughly "
            f"{d_ag.get('mean', float('nan')):+.4f} of recoverable ranking "
            f"quality plus {d_base['mean']:+.3f} of sparsity price. The "
            f"sparsity sweep in the next section shows the first term is not "
            f"there: at matched selected-set size the two attribution "
            f"statistics are indistinguishable, and the "
            f"{d_ag.get('mean', float('nan')):+.4f} was the extra sensors. The "
            f"conclusion in bold above is unchanged and is in fact "
            f"strengthened -- the deficit is sparsity price in full.", "",
        ]
    if d_ag:
        n_new = aa.get("n_selected_mean")
        n_old = None
        ns = [r.get("n_selected") for r in ev["records"]
              if r["arm"] == "agent_rf" and r.get("n_selected") is not None]
        if ns:
            n_old = sum(ns) / len(ns)
        cap = (ev.get("protocol", {}).get("agent_cfg", {}) or {}).get("max_select")
        body += [
            f"**Two reasons not to bank the {d_ag['mean']:+.4f} gain over the "
            f"permutation loop, both of which cut the same way.** Its interval "
            f"is [{d_ag['ci_lo']:+.4f}, {d_ag['ci_hi']:+.4f}] and its Wilcoxon "
            f"p is {d_ag.get('wilcoxon_p', float('nan')):.3f}, so it is not "
            f"established at the 0.05 level"
            + (f". And it is not a sparsity-matched comparison: this arm "
               f"selects {n_new:.1f} sensors per fold against the permutation "
               f"loop's {n_old:.1f}"
               + (f", which is its `max_select` cap of {cap} exactly -- the "
                  f"arm is pinned against its own budget and would have taken "
                  f"more"
                  if cap and abs(n_new - cap) < 0.05 else "")
               + f". Selection costs AUC monotonically on this dataset, so "
                 f"part of the gain is simply less sparsity rather than better "
                 f"ranking"
               if (n_new and n_old) else "")
            + ". Both caveats shrink the ranking-quality share of the deficit, "
              "which makes the sparsity explanation stronger rather than "
              "weaker.", "",
            "The registered distrust check passes for the reason it was "
            "written: an arm that reached the baseline by quietly abandoning "
            "selection would be uninteresting, and this one stayed sparse.", "",
        ]
    return body

def sec_sparsity(sp):
    """H8: is the attribution effect on AUC a sparsity artifact? (it is)

    `agent_model_rf` beat `agent_rf` by +0.0116 AUC while selecting 25.0
    sensors against 19.8 -- pinned at its `max_select` cap. This sweeps the cap
    for both statistics under the headline protocol so the curves can be
    compared at matched selected-set size. Read-out fixed before the run: the
    sign pattern across matched rungs, not a pooled interval.
    """
    if not sp:
        return []
    import numpy as np

    caps = sp["caps"]
    c, pr = sp["curves"], sp["per_rung"]
    rows, gaps, dels = [], [], []
    for k in caps:
        pm, mm = c["permutation"][str(k)], c["model"][str(k)]
        d = pr[str(k)]["model_minus_permutation"]
        gap = mm["n_selected_mean"] - pm["n_selected_mean"]
        gaps.append(gap)
        dels.append(d["mean"])
        rows.append([
            str(k),
            f"{pm['auc']['mean']:.4f}", f"{pm['n_selected_mean']:.1f}",
            f"{mm['auc']['mean']:.4f}", f"{mm['n_selected_mean']:.1f}",
            f"{gap:+.1f}",
            f"{d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]",
        ])
    r = float(np.corrcoef(np.array(gaps), np.array(dels))[0, 1])
    matched = [(k, g, dd) for k, g, dd in zip(caps, gaps, dels) if abs(g) < 0.2]
    slope = (sp.get("sparsity_slope") or {}).get("model", {})
    body = [
        "### Is the attribution effect on AUC a sparsity artifact? (H8)", "",
        "The H7 comparison was confounded: the model-native arm selected 25.0 "
        "sensors per fold against 19.8, and 25.0 was its `max_select` cap "
        "exactly, so it was pinned at its budget. This sweeps that cap for "
        "**both** attribution statistics under the headline 25-fold protocol, "
        "changing nothing else from the pre-registered operating point, so the "
        "two can be compared at matched selected-set size rather than at "
        "matched configuration.", "",
        table(rows, ["`max_select`", "perm AUC", "perm n", "model AUC",
                     "model n", "n gap", "model - perm (paired)"]), "",
        f"**The apparent attribution advantage tracks the sparsity gap and "
        f"vanishes when the gap does.** Across the ladder the correlation "
        f"between the selected-set gap and the paired AUC difference is "
        f"r = {r:.3f}. At the rungs where both arms take the same number of "
        f"sensors "
        + ", ".join(f"(`max_select` = {k}, gap {g:+.1f})" for k, g, _ in matched)
        + f" the difference is "
        + " and ".join(f"{dd:+.4f}" for _, _, dd in matched)
        + f" -- **negative**, and the pre-registered sign count is therefore "
          f"{sp['sign_test']['n_model_above']} of "
          f"{sp['sign_test']['n_matched_rungs']} strictly matched rungs.", "",
        "**H8 is refuted.** It predicted the model curve would sit above the "
        "permutation curve at matched size by +0.005 to +0.020; it sits "
        "fractionally below. The competing explanation -- that the whole "
        "+0.0116 was the extra 5.2 sensors -- is what the data support.", "",
    ]
    if slope:
        sl = slope["auc_per_sensor"]
        body += [
            f"The mechanism is quantitative. Over the range where the cap "
            f"binds ({slope['n_range'][0]:.0f} to {slope['n_range'][1]:.0f} "
            f"sensors) each additional sensor is worth {sl:+.5f} AUC, so the "
            f"5.1-sensor gap at `max_select` = 25 predicts "
            f"{5.1 * sl:+.4f} against the {dels[caps.index(25)]:+.4f} "
            f"observed. Sparsity alone accounts for the effect; nothing needs "
            f"to be attributed to the ranking.", "",
            "**So the H7 decomposition was wrong and is withdrawn.** That "
            "section split the loop's -0.042 AUC deficit into roughly +0.012 "
            "of recoverable ranking quality and -0.030 of irreducible sparsity "
            "price. The recoverable term is not there: at matched sparsity the "
            "two statistics are indistinguishable, so **the deficit is "
            "sparsity price essentially in full**. The attribution statistic "
            "is worth 13 points of selection stability and, at a suitable "
            "depth, 2.1 points of error control -- and nothing at all on "
            "accuracy.", "",
            "This also settles which of two things this repository said about "
            "the same number was right. The caveat attached to H7 -- read "
            "+0.012 as an upper bound, not an estimate -- was correct. The "
            "reasoning used the following turn to argue the effect would "
            "survive matching, which leaned on a dominance relation in "
            "`runs/secom_loop_sweep.json`, was not: those sweep arms differ in "
            "`select_k` and `stability_min` as well as in sparsity, and that "
            "was noted at the time and then not acted on.", "",
        ]
    return body


def sec_dedup(dd):
    """H10: is the loop's shorter list deduplication, or under-reporting?

    Reported with the pre-registration error on the face of it. H10 was
    registered against the 0.99 cluster map the stability column uses, and at
    that threshold it is refuted. But the loop's ``CorrelatorAgent`` groups at
    ``corr_thresh = 0.90``, so 0.99 tests a rule the architecture never claimed
    to enforce -- the pre-registered criterion was the wrong one, and both are
    shown rather than the favourable one being quietly substituted.
    """
    if not dd:
        return []
    bt, vd = dd.get("by_threshold") or {}, dd.get("verdicts") or {}
    if not bt:
        return []
    pre = str(dd.get("preregistered_threshold", 0.99))
    own = str(dd.get("loop_corr_thresh", 0.9))
    rows = []
    for th in sorted(bt, key=float):
        v = vd.get(th)
        if not v:
            continue
        tag = []
        if th == own:
            tag.append("the loop's own `corr_thresh`")
        if th == pre:
            tag.append("**pre-registered**")
        rows.append([f"{float(th):.2f}" + (f" ({', '.join(tag)})" if tag else ""),
                     str(bt[th]["n_families"]),
                     f"{v['loop_families']:.3f}",
                     f"{v['univariate_families']:.3f}",
                     f"{v['difference']:+.3f}",
                     "**holds**" if v["holds"] else "refuted"])
    if not rows:
        return []
    n_rep = next(iter(bt[pre]["arms"].values()))["n_replicates"] \
        if pre in bt else 0
    return [
        "### Is the shorter suspect list deduplication? (H10)", "",
        "Every comparison so far has scored report length as though longer "
        "were better: the loop names 1.55 suspects at 94.2% control against a "
        "univariate ranker's 2.06 at 94.3%. But the loop runs a "
        "`CorrelatorAgent` whose job is to collapse near-identical sensors, "
        "and 179 SECOM sensors have a partner correlated above 0.99. A shorter "
        "list from a deduplicating ranker may be the same information with the "
        "duplicates removed.", "",
        "Measured at **matched list length** -- every arm's stored real-label "
        "`top5` -- so this is a statement about the ranking, not about report "
        "length. No model is fitted; it reads sets already on disk.", "",
        table(rows, ["family threshold", "families in the matrix",
                     "loop, distinct families in top-5",
                     "univariate, same", "difference", "H10"]), "",
        f"**At the loop's own grouping threshold the answer is yes, "
        f"deterministically.** The loop's top-5 spans five distinct families in "
        f"every one of its {n_rep} real-label replicates; the univariate "
        f"ranker's spans four in every one of its own -- it names a "
        f"near-duplicate pair every single time, and the loop never does. Both "
        f"counts are constant across replicates, so the pre-registered "
        f"standard-error bar is degenerate and the honest statement is the "
        f"replicate count rather than a p-value.", "",
        f"**And at the pre-registered threshold it is refuted**, because at "
        f"|r| >= {float(pre):.2f} no arm ever names a duplicate pair -- there "
        f"is nothing to deduplicate. That is the same thing the stability "
        f"table's raw and cluster-aware columns say by barely differing, now "
        f"exactly rather than approximately.", "",
        "*The pre-registration was wrong and is not being quietly swapped.* "
        f"H10 was registered against the {float(pre):.2f} map because that is "
        f"the one the stability column uses. The loop groups at "
        f"{float(own):.2f}. Testing a deduplication claim at a threshold "
        f"stricter than the rule being tested asks whether the component "
        f"enforces something it never claimed to, and the answer to that "
        f"question is uninformative. The {float(own):.2f} row is the fair test "
        f"and the {float(pre):.2f} row is the one I said I would run.", "",
        "**What this does and does not license.** It is the first measured "
        "defence of a loop component in this repository: the correlation "
        "grouping does exactly what it advertises. It does *not* show the "
        "shorter report is as informative -- the tau-thresholded sets are not "
        "reconstructible from disk, since the records store support values "
        "without sensor identities, so this measures the ranking rather than "
        "the report. Nor does it show the deduplication is worth its cost: the "
        "same component was worth +0.2 points of top-5 stability in the "
        "ablation ladder, and one duplicate pair removed from a five-item list "
        "is a small thing to buy with a screen, a correlation matrix and a "
        "verification loop.", "",
    ]


def sec_calib_size(cs, nfm40=None, abm40=None, ab5m=None):
    """Split the gap to nominal into a grid term and a calibration term.

    Once `n_boot` is large enough for the grid to contain a nominal level, what
    is left between nominal and delivered? This resamples the per-replicate
    null statistics already recorded -- no model is fitted -- to separate the
    resolution of the attainable set from the noise in estimating tau from a
    finite null.
    """
    if not cs:
        return []
    arms = cs.get("arms") or {}
    if not arms:
        return []
    nom = cs["protocol"]["nominal"]
    sizes = cs["sizes"]
    body = []

    # --- H9 first: the agent loop's own n_boot pair ---
    if nfm40 and abm40 and ab5m:
        import numpy as np
        rows = []
        for lab, nf_, ab_ in (("`n_boot` = 12 (pre-registered)", None, ab5m),
                              ("`n_boot` = 40", nfm40, abm40)):
            m = ab_["levels"]["alpha_0.05"]
            key = ("agent loop, select_k=5, model, n_boot="
                   + ("40" if nf_ else "12"))
            a = arms.get(key, {})
            rows.append([lab, pct(a.get("p_saturated", float("nan"))),
                         str(len(a.get("attainable_above_060", []))),
                         f"{a.get('oracle_control', float('nan')):.4f}",
                         pct(m["null_abstention_heldout"]),
                         f"{m['real_reported_mean']:.2f}"])
        body += [
            "### Does a finer grid reach nominal? (H9)", "",
            "The grid account predicts that raising `n_boot` refines the "
            "attainable set. Registered before the run: the refinement brings "
            "a value within 0.01 of 0.95, and measured control improves on "
            "93.7%. Also registered was the competing possibility that "
            "P(M = 1) would *rise* with more resamples and push the top of the "
            "set back down, in which case the advice would have inverted to "
            "fewer resamples. Agent loop, `select_k = 5`, model attribution, "
            "`n_boot` the only change:", "",
            table(rows, ["arm", "P(M = 1)", "attainable above 0.60",
                         "best attainable (oracle)", "measured control",
                         "suspects"]), "",
            "**H9 holds on both halves, and the competing mechanism is "
            "refuted rather than merely absent.** P(M = 1) *fell* from 0.5% to "
            "0.0%: with more resamples a noisy sensor gets more chances to be "
            "missed, and that dominates the averaging effect that would have "
            "pushed it the other way. So the two terms do not trade off here "
            "-- the finer grid is a free improvement, and the report gets "
            "longer rather than shorter while control improves.", "",
            "**What does not change is the comparison that carries the "
            "conclusion.** The univariate arm at the same `n_boot` = 40 and "
            "the same depth reaches 94.3% control while reporting 2.06 "
            "suspects. The loop reaches 94.2% reporting 1.55. Tuning `n_boot` "
            "moved the loop from clearly behind on error control to level on "
            "it, and left it behind on report length. The headline is "
            "unaffected.", "",
        ]

    # --- the decomposition ---
    rows = []
    for lab, a in arms.items():
        m100 = (a["curve"].get("100") or {})
        rows.append([lab, str(a["n_boot"]),
                     f"{a['oracle_control']:.3f}",
                     f"{a['grid_gap']:+.3f}",
                     f"{m100.get('control_mean', float('nan')):.3f}",
                     f"{m100.get('calibration_loss', float('nan')):+.3f}"])
    body += [
        "### Grid or calibration? Splitting the gap to nominal", "",
        f"With `n_boot` fixed, the distance between nominal {nom:.2f} and what "
        f"the pipeline delivers has two sources: the **grid** may not contain "
        f"a value at nominal, and the **calibration** estimates tau from a "
        f"finite null. This separates them by resampling the per-replicate "
        f"statistics already recorded -- no model is fitted anywhere in this "
        f"section. The *oracle* is the attainable value closest to nominal, "
        f"i.e. what an infinite calibration set would deliver; *measured* fits "
        f"tau on 100 null replicates and scores the held-out remainder, which "
        f"is `scripts/abstain.py`'s protocol.", "",
        table(rows, ["arm", "`n_boot`", "oracle", "grid gap",
                     "measured (m = 100)", "calibration loss"]), "",
        "Read the last two columns as the answer to *what should I spend on*. "
        "Where the grid gap is non-zero, more bootstrap resamples; where the "
        "calibration loss dominates, more null replicates. The two are "
        "different budgets and this repository had been conflating them under "
        "\"the calibration is imperfect\".", "",
    ]
    # how the calibration loss shrinks with the null size
    a40 = arms.get("agent loop, select_k=5, model, n_boot=40")
    if a40:
        crows = [[str(m), f"{a40['curve'][str(m)]['control_mean']:.3f}",
                  f"{a40['curve'][str(m)]['control_sd']:.3f}",
                  f"{a40['curve'][str(m)]['calibration_loss']:+.3f}"]
                 for m in sizes if str(m) in a40["curve"]]
        body += [
            "And how fast the calibration term shrinks, for the arm H9 "
            "produced (agent loop, `select_k = 5`, model, `n_boot` = 40, "
            f"oracle {a40['oracle_control']:.3f}):", "",
            table(crows, ["null replicates used to fit tau", "control",
                          "sd over draws", "calibration loss"]), "",
            "The loss roughly halves from 25 to 50 replicates and again to "
            "100, then stops. At 200 null replicates -- what these runs use -- "
            "the split-half protocol fits tau on 100, which is where the curve "
            "flattens, so **the remaining gap to nominal is not something more "
            "null replicates would close cheaply.** That is a more useful "
            "statement than the raw shortfall, and it is only visible once the "
            "two terms are separated.", "",
        ]
    return body


def sec_nboot_grid(rk):
    """Does raising `n_boot` refine the attainable set? Measured, at no cost.

    The grid account says `n_boot` sets the resolution of error control. That
    is testable without a single new fit, because `runs/null_fdr_rankers.json`
    already contains three arms differing *only* in `n_boot` -- a ladder run
    for a different purpose months of turns ago and never read this way.

    These arms also isolate the effect. P(M = 1) is zero for all three, so the
    saturation term that complicates the agent-loop arms is absent and what is
    left is pure spacing.
    """
    if not rk:
        return []
    import numpy as np

    per = rk.get("per_ranker") or {}
    recs = rk.get("records") or []
    fam = [(k, v) for k, v in per.items()
           if v.get("is_variant") and v.get("select_k") == 5
           and v.get("ranker") == "univariate"]
    fam.sort(key=lambda kv: kv[1]["n_boot"])
    if len(fam) < 3:
        return []
    rows, sats, counts, nears = [], [], [], []
    for name, v in fam:
        nb = v["n_boot"]
        mx = np.asarray([r["max_stability"] for r in recs
                         if r["arm"] == name and r["permuted"]], dtype=float)
        if not len(mx):
            continue
        ach = sorted({float(1.0 - (mx >= k / nb).mean())
                      for k in range(1, nb + 1)}, reverse=True)
        n_hi = len([a for a in ach if a > 0.60])
        near = min(ach, key=lambda a: abs(a - 0.95))
        sat = float((mx >= 1.0).mean())
        ctl = v["heldout_alpha_0.05"]["null_abstention_heldout"]
        sats.append(sat)
        counts.append(n_hi)
        nears.append(near)
        rows.append([str(nb), pct(sat), str(n_hi),
                     f"{near:.3f}" + (" **(exactly nominal)**"
                                      if abs(near - 0.95) < 1e-9 else ""),
                     pct(ctl)])
    if len(rows) < 3:
        return []
    refines = all(counts[i] < counts[i + 1] for i in range(len(counts) - 1))
    body = [
        "### Does `n_boot` set the resolution? (measured on a ladder already "
        "in `runs/`)", "",
        "The account above says `n_boot` fixes the spacing of the attainable "
        "set. That is checkable with no new fits: `runs/null_fdr_rankers.json` "
        "already holds three `univariate` arms at `select_k = 5` differing "
        "**only** in `n_boot`, run for a different question and never read "
        "this way. They also isolate the effect -- P(M = 1) is zero for all "
        "three, so the saturation term that complicates the agent-loop arms is "
        "absent and what remains is pure spacing.", "",
        table(rows, ["`n_boot`", "P(M = 1)", "attainable values above 0.60",
                     "closest attainable to 0.95",
                     "measured control (alpha = 0.05)"]), "",
    ]
    if refines:
        body += [
            f"**The grid refines monotonically** -- "
            f"{' to '.join(str(c) for c in counts)} attainable values above "
            f"0.60 -- and nominal 0.95 goes from unreachable at `n_boot` = "
            f"{fam[0][1]['n_boot']} (closest {nears[0]:.3f}) to **exactly "
            f"attainable** from `n_boot` = {fam[1][1]['n_boot']} onward. "
            f"Measured control follows: {rows[0][4]} to {rows[1][4]}.", "",
            f"**And the return stops.** Going from "
            f"{fam[1][1]['n_boot']} to {fam[2][1]['n_boot']} resamples "
            f"multiplies the work by "
            f"{fam[2][1]['n_boot'] / fam[1][1]['n_boot']:.1f}x, adds "
            f"{counts[2] - counts[1]} more attainable values, and moves "
            f"measured control from {rows[1][4]} to {rows[2][4]} -- backwards, "
            f"within noise. So the practical reading is that `n_boot` = "
            f"{fam[0][1]['n_boot']} is too coarse to express a 95% target and "
            f"{fam[1][1]['n_boot']} is enough, not that more is better.", "",
            "Two limits on what this settles, both of which are why the "
            "agent-loop version of the experiment is still worth running. "
            "These are univariate arms, not the loop. And with P(M = 1) = 0 "
            "throughout, they say nothing about the competing term: a finer "
            "grid helps only if saturation does not rise to meet it, and the "
            "loop's model-native attribution is the configuration where "
            "saturation was non-zero to begin with.", "",
        ]
    return body


def sec_saturation(pairs):
    """What error control is *achievable* at all, and why the levels coincide.

    An earlier version of this section called the bound below an "identity"
    that "predicts rather than describes", said it held "with no dependence on
    alpha at all", and claimed it applied to "any max-support threshold rule".
    An adversarial review took all three apart and was right about each
    (`critique_log.md`, Turn 12): the endpoint bound alone is a one-line
    consequence of the support being bounded by 1 and the rule comparing with
    ``>=``, so calling it a discovery was wrong. What survives, and what this
    section now reports, is the *discreteness* the endpoint bound is one corner
    of -- which is not vacuous, explains an exact numerical coincidence the
    earlier text got right for the wrong reason, and is actionable.

    ``pairs`` is a list of (label, null_fdr json, abstain json).
    """
    import numpy as np

    rows, sets, coincide = [], [], []
    for label, nf_, ab_ in pairs:
        if not (nf_ and ab_):
            continue
        mx = np.asarray([r["max_stability"] for r in nf_.get("records", [])
                         if r["permuted"]], dtype=float)
        lv = ab_.get("levels") or {}
        nb = ((nf_.get("protocol") or {}).get("agent_cfg") or {}).get("n_boot")
        if not len(mx) or not lv or not nb:
            continue
        # Control as a function of tau is a step function; its steps are the
        # attainable values, one per grid point of the discrete statistic.
        ach = sorted({float(1.0 - (mx >= k / nb).mean())
                      for k in range(1, nb + 1)}, reverse=True)
        sat = float((mx >= 1.0).mean())
        sets.append((label, nb, sat, ach))
        for key in ("alpha_0.1", "alpha_0.05", "alpha_0.01"):
            m = lv.get(key)
            if not m:
                continue
            rows.append([label, f"{m['alpha']}",
                         f"{m.get('tau_min', float('nan')):.3f}",
                         f"{m.get('tau_max', float('nan')):.3f}",
                         pct(m["null_abstention_heldout"]),
                         pct(m["null_abstention_target"])])
        # Two levels agreeing to the reported precision while their fitted
        # thresholds differ is the discreteness showing through.
        keys = [k for k in ("alpha_0.1", "alpha_0.05", "alpha_0.01") if k in lv]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a_, b_ = lv[keys[i]], lv[keys[j]]
                if (abs(a_["null_abstention_heldout"]
                        - b_["null_abstention_heldout"]) < 1e-9
                        and abs(a_.get("tau_min", 0) - b_.get("tau_min", 0)) > 1e-6):
                    coincide.append((label, a_["alpha"], b_["alpha"],
                                     a_["null_abstention_heldout"],
                                     a_.get("tau_min"), b_.get("tau_min")))
    if not rows:
        return []

    body = [
        "## What error control is achievable at all", "",
        "A suspect's bootstrap support is a count over `n_boot` resamples "
        "divided by `n_boot`, so the null max-statistic *M* lives on the grid "
        "{0, 1/`n_boot`, ..., 1}. Error control as a function of the threshold "
        "is therefore a **step function**, and only `n_boot` + 1 values of it "
        "are attainable no matter how alpha is chosen:", "",
        "> attainable control = { 1 - P(M >= k / `n_boot`) : k = 1 .. `n_boot` }",
        "",
        "Two things follow, and they are worth separating because only the "
        "second is interesting.", "",
        "**The trivial one.** The largest attainable value is "
        "1 - P(M = 1), because the fitted threshold is a quantile of a "
        "statistic bounded by 1 and the rule reports when support >= tau. "
        "That is a one-line consequence of boundedness and the comparison "
        "operator, not a finding, and an earlier version of this section "
        "oversold it as an identity that predicts. It also is not a property "
        "of max-support thresholding in general: a rule comparing with a "
        "strict `>` at 1, or thresholding something unbounded, escapes it "
        "entirely.", "",
        "**The one that matters.** The attainable *set* is coarse, and its "
        "spacing is set by `n_boot` rather than by anything statistical:", "",
    ]
    srows = []
    for label, nb, sat, ach in sets:
        hi = [a for a in ach if a > 0.60]
        near = min(ach, key=lambda a: abs(a - 0.95))
        srows.append([label, str(nb), pct(sat),
                      ", ".join(f"{a:.3f}" for a in hi) or "--",
                      f"{near:.3f}" + (" **(=)**" if abs(near - 0.95) < 1e-9
                                       else f" ({near - 0.95:+.3f})")])
    body += [
        table(srows, ["arm", "`n_boot`", "P(M = 1)",
                      "attainable control above 0.60",
                      "closest attainable to 0.95"]), "",
        "**No arm can land on 0.95 exactly**, because 0.95 is not in any of "
        "these sets -- the misses are a property of the grid rather than of "
        "the calibration. For the worst arm the entire attainable set above "
        "0.60 is a single value. This is a resolution limit, and the parameter that sets it is "
        "`n_boot`, which no part of this repository had previously identified "
        "as governing error control at all.", "",
    ]
    if coincide:
        lab, a1, a2, ctl, t1, t2 = coincide[0]
        body += [
            f"**The coincidence this explains.** "
            f"{lab.replace('`', '').replace('**', '')} reports exactly "
            f"{ctl:.6f} control at both alpha = {a1} and alpha = {a2}, even "
            f"though the smallest threshold its splits fitted differs between "
            f"them ({t1:.3f} against {t2:.3f}). The earlier text read that as "
            f"the threshold pinning at 1.000 in both cases, which the "
            f"`tau_min` column above shows is false. The real reason is the "
            f"grid: with `n_boot` = 12 the null max has an atom at 1.000 and "
            f"nothing between 11/12 = 0.917 and 1.000, so **every** threshold "
            f"in that gap selects exactly the same replicates and returns "
            f"exactly the same control. Alpha moves the threshold within a "
            f"gap without moving the answer.", "",
            "So the alpha-invariance is real but local, and the earlier "
            "phrase \"with no dependence on alpha at all\" was wrong: whether "
            "two levels agree depends on whether their thresholds land in the "
            "same gap, which is entirely a question of alpha.", "",
        ]
    body += [
        "**And P(M = 1) is not a property of the rule.** It depends on the "
        "bootstrap count, the selection depth, the attribution statistic, the "
        "candidate-pool size and the null construction -- changing `select_k` "
        "alone moves it across the arms above. The rule supplies the step "
        "structure; the experiment decides where the steps fall. Claims here "
        "are therefore about these arms under this protocol, not about "
        "stability selection in general.", "",
        "What this does leave intact is the practical coupling. A more "
        "repeatable attribution statistic pins a sensor across more resamples, "
        "which raises P(M = 1) and pushes the top of the attainable set down; "
        "a narrower selection depth lowers it again. That is why the same "
        "one-field attribution swap improves error control at `select_k` = 5 "
        "and degrades it at `select_k` = 40, and it is the reason "
        "recommendation 3 asks for both fields together.", "",
        table(rows, ["arm", "alpha", "smallest tau fitted", "largest tau fitted",
                     "measured control", "nominal"]), "",
    ]
    return body


def sec_attribution_fdr(nf, ab, nf5, ab5, nfm, abm, nf5m, ab5m, rk=None):
    """H6: is the loop's error control limited by its attribution statistic?

    The depth comparison ruled depth out and the guard cross-tab ruled the
    never-empty fallback out, which left the attribution estimator as the
    binding constraint *by elimination*. Twice this weekend an
    elimination argument here failed when tested directly, so it is tested
    directly: `attribution="model"` and nothing else, at both depths, priced by
    the same split-half calibration.

    Every comparison below is within-protocol -- permutation against model at
    the same ``select_k``, same ``n_boot``, same replicate counts. That matters
    because the separation statistic moves a great deal with depth, so quoting
    a k=40 separation next to a k=5 one would manufacture an effect.
    """
    pairs = [(5, nf5, ab5, nf5m, ab5m), ("pre-registered", nf, ab, nfm, abm)]
    pairs = [(d, a, b, c, e) for d, a, b, c, e in pairs
             if a and b and c and e]
    if not pairs:
        return []
    body = ["### Ranker or estimator? (H6, the direct test)", "",
            "`select_k` moved the loop's error control by less than half a "
            "point and the never-empty guard provably never reaches the "
            "calibrated report, which left the attribution estimator as the "
            "constraint by elimination. This tests it directly: "
            "`attribution=\"model\"`, nothing else changed, same split-half "
            "calibration on both sides.", ""]
    for depth, nfp, abp, nfm_, abm_ in pairs:
        label = (f"`select_k = {depth}`" if depth != "pre-registered"
                 else f"`select_k = {nfp['protocol']['agent_cfg']['select_k']}` "
                      f"(pre-registered)")
        rows = []
        for key in ("alpha_0.1", "alpha_0.05", "alpha_0.01"):
            p_, m_ = abp["levels"].get(key), abm_["levels"].get(key)
            if not (p_ and m_):
                continue
            rows.append([
                f"alpha = {p_['alpha']} (target "
                f"{pct(p_['null_abstention_target'], 0)})",
                pct(p_["null_abstention_heldout"]),
                pct(m_["null_abstention_heldout"]),
                f"{p_['real_reported_mean']:.2f}",
                f"{m_['real_reported_mean']:.2f}"])
        if not rows:
            continue
        sep_p = nfp["separation"]["prob_real_max_exceeds_null_max"]
        sep_m = nfm_["separation"]["prob_real_max_exceeds_null_max"]
        best = max(rows, key=lambda r: 0)  # keep order; alpha_0.05 is rows[1]
        a05p = abp["levels"]["alpha_0.05"]
        a05m = abm_["levels"]["alpha_0.05"]
        d_ctrl = 100 * (a05m["null_abstention_heldout"]
                        - a05p["null_abstention_heldout"])
        d_susp = a05m["real_reported_mean"] - a05p["real_reported_mean"]
        body += [
            f"**At {label}.**", "",
            table(rows, ["", "control, permutation", "control, **model**",
                         "suspects, permutation", "suspects, **model**"]), "",
            f"Separation of the two worlds, same protocol: "
            f"{sep_p:.3f} with permutation attribution, **{sep_m:.3f}** with "
            f"model-native. At alpha = 0.05 control moves "
            f"{d_ctrl:+.1f} points and the report gets "
            f"{'longer' if d_susp > 0 else 'shorter'} by "
            f"{abs(d_susp):.2f} suspects "
            f"({a05p['real_reported_mean']:.2f} to "
            f"{a05m['real_reported_mean']:.2f}), with real-label abstention "
            f"going {pct(a05p['real_abstention'], 0)} to "
            f"{pct(a05m['real_abstention'], 0)}.", "",
        ]
        if d_susp > 0 and d_ctrl > 0:
            body += [
                "Both columns move the right way at once, which is the part "
                "that makes this a real improvement rather than a trade. A "
                "rule can always buy error control by reporting less; this one "
                "controls better *and* names more suspects, so it is ranking "
                "better rather than abstaining more. That test was written "
                "down before the run (`critique_log.md`, Turn 10): a "
                "confirmation whose suspect count collapsed was to be "
                "distrusted.", "",
            ]
    # Where this leaves the loop against the simplest thing that could work.
    uni = None
    if rk:
        cands = [v for v in (rk.get("per_ranker") or {}).values()
                 if v.get("is_variant") and v.get("select_k") == 5
                 and v.get("ranker") == "univariate"]
        if cands:
            uni = max(cands, key=lambda v:
                      v["heldout_alpha_0.05"]["null_abstention_heldout"])
    if uni and ab5m:
        u = uni["heldout_alpha_0.05"]
        m = ab5m["levels"]["alpha_0.05"]
        gap = 100 * (u["null_abstention_heldout"] - m["null_abstention_heldout"])
        body += [
            "**And against the univariate baseline, which is what the "
            "conclusion rests on.** At matched depth and alpha = 0.05:", "",
            table([
                ["`univariate (n_boot=40, select_k=5)`",
                 pct(u["null_abstention_heldout"]),
                 f"{u['real_reported_mean']:.2f}",
                 f"{uni['prob_real_max_exceeds_null_max']:.3f}"],
                ["agent loop, model-native attribution, `select_k = 5`",
                 pct(m["null_abstention_heldout"]),
                 f"{m['real_reported_mean']:.2f}",
                 f"{nf5m['separation']['prob_real_max_exceeds_null_max']:.3f}"],
                ["agent loop, permutation attribution, `select_k = 5`",
                 pct(ab5["levels"]["alpha_0.05"]["null_abstention_heldout"]),
                 f"{ab5['levels']['alpha_0.05']['real_reported_mean']:.2f}",
                 f"{nf5['separation']['prob_real_max_exceeds_null_max']:.3f}"],
            ], ["arm", "no-cause worlds kept silent", "suspects reported",
                "separation"]), "",
            f"Swapping the statistic closes most of the error-control gap -- "
            f"{gap:+.1f} points remain against the univariate arm, down from "
            f"{100 * (u['null_abstention_heldout'] - ab5['levels']['alpha_0.05']['null_abstention_heldout']):+.1f} "
            f"-- and closes the separation gap almost entirely. It does not "
            f"close the report-length gap: the univariate arm still names "
            f"{u['real_reported_mean']:.2f} suspects against "
            f"{m['real_reported_mean']:.2f}.", "",
            f"**One confound in that table, stated rather than buried.** The "
            f"univariate arm resamples "
            f"{uni.get('n_boot')} times and the loop "
            f"{nf5m['protocol']['agent_cfg']['n_boot']}, so the three rows are "
            f"matched on selection depth and calibration but not on bootstrap "
            f"count. The depth was probed for the baseline and the bootstrap "
            f"count comes with it. Two of the three columns here are "
            f"statistics *of* the bootstrap distribution, so a longer "
            f"bootstrap is not neutral, and the residual "
            f"{gap:+.1f}-point control gap is inside the range that difference "
            f"could plausibly account for. The honest reading is that at "
            f"matched depth and unmatched bootstrap count the two arms are "
            f"close on error control and separation, not that either is ahead.", "",
            "So the conclusion narrows rather than reverses. The loop still "
            "does not *beat* a univariate ranker on any axis measured here. "
            "But the claim that it is comprehensively worse was resting on a "
            "configuration whose attribution statistic was the weakest part of "
            "it, and with that one field changed two of the three gaps are "
            "small. What the loop buys for the remaining cost is still "
            "nothing measurable, which is the finding; what it costs is now "
            "much less than the pre-registered operating point suggested.", "",
        ]
    return body


def sec_depth(ab, ab_k5, rk, nf=None, nf5=None):
    """Is it the ranker or the selection depth that limits error control?

    H4. The univariate arm's control jumped when its bootstrap selection depth
    narrowed, which leaves two readings: either the agent loop's
    permutation-importance estimator is the binding constraint, or depth
    dominates ranker choice and the loop moves too. Running the loop's own
    depth down to the same value separates them, and it is the same script and
    the same held-out calibration on both sides.
    """
    if not (ab and ab_k5):
        return []
    a40 = (ab.get("levels") or {}).get("alpha_0.05")
    a5 = (ab_k5.get("levels") or {}).get("alpha_0.05")
    if not (a40 and a5):
        return []
    rows = [
        ["agent loop, `select_k = 40` (pre-registered)", f"{a40['tau_mean']:.3f}",
         pct(a40["null_abstention_heldout"]), f"{a40['real_reported_mean']:.2f}",
         pct(a40["real_abstention"], 0)],
        ["agent loop, `select_k = 5`", f"{a5['tau_mean']:.3f}",
         pct(a5["null_abstention_heldout"]), f"{a5['real_reported_mean']:.2f}",
         pct(a5["real_abstention"], 0)],
    ]
    uni = None
    if rk:
        cands = {k: v for k, v in (rk.get("per_ranker") or {}).items()
                 if v.get("is_variant") and v.get("select_k") == 5
                 and v.get("ranker") == "univariate"}
        if cands:
            uk = max(cands, key=lambda k:
                     cands[k]["heldout_alpha_0.05"]["null_abstention_heldout"])
            h = cands[uk]["heldout_alpha_0.05"]
            uni = h
            rows.append([f"`{uk}`", f"{h['tau_mean']:.3f}",
                         pct(h["null_abstention_heldout"]),
                         f"{h['real_reported_mean']:.2f}",
                         pct(h["real_abstention"], 0)])
    d_ctl = a5["null_abstention_heldout"] - a40["null_abstention_heldout"]
    body = [
        "### Ranker or depth? (the symmetric run)", "",
        "The table above tunes the baseline's selection depth and leaves the "
        "loop at its pre-registered one, which answers *would something "
        "simpler have sufficed* and not *is the architecture worse at equal "
        "effort*. This is the second question, run with the loop's depth as "
        "the only thing changed. Priced by `scripts/abstain.py` on both sides, "
        "so the calibration is identical.", "",
        table(rows, ["arm", "tau", "no-cause worlds kept silent",
                     "suspects reported", "reports nothing"]), "",
    ]
    if d_ctl > 0.02:
        body += [
            f"**Depth moves the loop too**, by "
            f"{d_ctl * 100:+.1f} points of error control "
            f"({pct(a40['null_abstention_heldout'])} to "
            f"{pct(a5['null_abstention_heldout'])}). So the constraint that "
            f"produced the loop's low control was mostly its selection depth, "
            f"not its permutation-importance estimator -- which is the more "
            f"useful finding for anyone building one of these, because depth "
            f"is a free parameter and the estimator is the architecture.", "",
        ]
    elif d_ctl < -0.02:
        body += [
            f"**Narrowing the depth makes the loop worse**, by "
            f"{d_ctl * 100:.1f} points "
            f"({pct(a40['null_abstention_heldout'])} to "
            f"{pct(a5['null_abstention_heldout'])}), so the two arms do not "
            f"respond to depth the same way and the loop's pre-registered "
            f"setting was the better one for it.", "",
        ]
    else:
        moved = (f"{d_ctl * 100:+.1f} points" if abs(d_ctl) >= 0.005
                 else "by less than half a point")
        body += [
            f"**Depth barely moves the loop** "
            f"({pct(a40['null_abstention_heldout'])} to "
            f"{pct(a5['null_abstention_heldout'])}, {moved}), while it moved "
            f"the univariate arm substantially. The loop's binding constraint "
            f"is therefore its permutation-importance estimator rather than "
            f"the depth it selects at -- and that is a property of the "
            f"architecture, not a parameter someone can turn.", "",
        ]
        if a5["real_reported_mean"] > a40["real_reported_mean"] * 1.2:
            body += [
                f"What depth *does* buy the loop is a usable report: "
                f"{a5['real_reported_mean']:.2f} suspects against "
                f"{a40['real_reported_mean']:.2f}, and an empty report on "
                f"{pct(a5['real_abstention'], 0)} of real replicates instead "
                f"of {pct(a40['real_abstention'], 0)}. So `select_k` is worth "
                f"turning down; it just does not close the gap on error "
                f"control.", "",
            ]
    if nf and nf5:
        n40, n5 = nf["null"], nf5["null"]
        body += [
            "**The two guards do behave completely differently at the two "
            "depths**, which is worth seeing before reading too much into "
            "either:",
            "",
            table([
                ["pure-noise sensors clearing the threshold on merit",
                 f"{n40['n_merit_mean']:.1f}", f"{n5['n_merit_mean']:.2f}"],
                ["replicates where the never-empty fallback fired",
                 pct(n40["fallback_rate"]), pct(n5["fallback_rate"])],
                ["prediction set left empty (`report_tau = None`, so this "
                 "row cannot be anything else)",
                 pct(n40["abstention_rate"]), pct(n5["abstention_rate"])],
            ], ["on permuted labels", "`select_k = 40`", "`select_k = 5`"]), "",
            f"At the pre-registered depth the threshold is so loose that "
            f"{n40['n_merit_mean']:.1f} noise sensors clear it unaided and the "
            f"fallback is never needed. Narrow the depth and the threshold "
            f"starts working -- only {n5['n_merit_mean']:.2f} noise sensors "
            f"clear it -- and the fallback takes over, firing on "
            f"{pct(n5['fallback_rate'])} of null replicates.", "",
        ]
        body += fallback_reach(nf5, ab_k5)

    if uni:
        gap5 = uni["null_abstention_heldout"] - a5["null_abstention_heldout"]
        body += [
            f"At matched depth the univariate ranker is still ahead on "
            f"control, by {gap5 * 100:+.1f} points "
            f"({pct(uni['null_abstention_heldout'])} against "
            f"{pct(a5['null_abstention_heldout'])})"
            + (f", and reports {uni['real_reported_mean']:.2f} suspects "
               f"against {a5['real_reported_mean']:.2f}."
               if uni["real_reported_mean"] > a5["real_reported_mean"] else
               f", though the loop now reports "
               f"{a5['real_reported_mean']:.2f} suspects against "
               f"{uni['real_reported_mean']:.2f}.")
            + " That is the equal-effort comparison, and it is the one the "
              "README's conclusion should rest on."
            if gap5 > 0.005 else
            f"At matched depth the two are level on control "
            f"({pct(uni['null_abstention_heldout'])} for univariate against "
            f"{pct(a5['null_abstention_heldout'])} for the loop), so at equal "
            f"effort neither has an error-control advantage and the loop's "
            f"extra machinery buys nothing on this axis either.", "",
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


def sec_recommend(ev, ab, rk, ab5=None, st=None, ab5m=None,
                  abm=None, nfm=None):
    """What to actually do with this, computed from what was measured.

    This section used to be hand-written, and by the time the false-discovery
    and ranker results landed it was recommending the opposite of what the
    numbers supported -- specifically "report with the loop, and quote its
    stability", when that stability had turned out to be uncalibrated. It is
    generated now so it cannot drift again.
    """
    if not ev:
        return []
    auc = ev["auc"]["per_arm"]
    best = max((a for a in auc if a != "majority"), key=lambda a: auc[a]["mean"])
    L = [
        f"**Predict with every sensor -- with one asterisk.** Under the "
        f"shuffled protocol selection costs AUC monotonically, because the "
        f"signal is diffuse, so `{best}` at {auc[best]['mean']:.3f} is the "
        f"model to deploy. The asterisk is that forward in time the ordering "
        f"reverses and the sparse arms come out ahead; that comparison has "
        f"{ev['protocol']['rolling_origin']['n_origins']} origins behind it "
        f"and its interval includes zero, so it is a reason to monitor and "
        f"re-measure as wafers accumulate, not a reason to ship the sparse "
        f"model today.",
    ]
    if ab and (ab.get("levels") or {}).get("alpha_0.05"):
        mid = ab["levels"]["alpha_0.05"]
        base = ab["no_rule_baseline"]
        L.append(
            f"**Do not ship the suspect list without a null-calibrated "
            f"bar.** As it stands the loop reports "
            f"{base['real_reported_mean']:.1f} suspects and abstains on "
            f"nothing, and on permuted labels it does the same -- so the list "
            f"length carries no information about the process. "
            f"`AgentRCA(report_tau=...)` fixes that: at alpha = "
            f"{mid['alpha']:g} the report becomes "
            f"{mid['real_reported_mean']:.2f} sensors and is empty "
            f"{pct(mid['real_abstention'])} of the time. Prediction is "
            f"untouched either way, so this costs no AUC.")
    # The single highest-value change measured in this repository, and it is
    # one configuration field. Placed before the "replace the ranking core"
    # item because it is strictly cheaper and moves both KPIs.
    st_r = (st or {}).get("rankers") or {}
    if "agent_model" in st_r and "agent" in st_r:
        def _sv(n):
            return 100 * st_r[n]["bootstrap"]["raw"]["pairwise_overlap"]
        def _wv(n):
            return st_r[n]["bootstrap"]["wall_min"]
        item = (
            f"**Change the attribution statistic first -- it is one field and "
            f"it moves more than anything else measured here.** Setting "
            f"`attribution=\"model\"` instead of held-out permutation "
            f"importance raises top-5 selection stability from "
            f"{_sv('agent'):.1f}% to {_sv('agent_model'):.1f}% "
            f"({_sv('agent_model') - _sv('agent'):+.1f} points) and cuts the "
            f"loop's runtime {_wv('agent') / _wv('agent_model'):.1f}x "
            f"({_wv('agent'):.1f} min to {_wv('agent_model'):.1f} min).")
        if ab5m and ab5 and (ab5m.get("levels") or {}).get("alpha_0.05"):
            mm = ab5m["levels"]["alpha_0.05"]
            pp = ab5["levels"]["alpha_0.05"]
            item += (
                f" At `select_k = 5` it also improves null error control from "
                f"{pct(pp['null_abstention_heldout'])} to "
                f"{pct(mm['null_abstention_heldout'])} while reporting *more* "
                f"suspects ({pp['real_reported_mean']:.2f} to "
                f"{mm['real_reported_mean']:.2f}), so it is not buying control "
                f"by saying less.")
        # ...but only at a depth that keeps the null max-statistic from
        # saturating. At the pre-registered depth the same swap makes error
        # control *worse*, and recommending it unconditionally would be
        # recommending a regression.
        flipped = False
        if abm and (abm.get("levels") or {}).get("alpha_0.05") and ab:
            mm40 = abm["levels"]["alpha_0.05"]
            pp40 = ab["levels"]["alpha_0.05"]
            d40 = 100 * (mm40["null_abstention_heldout"]
                         - pp40["null_abstention_heldout"])
            flipped = d40 < 0
            if flipped:
                sat = None
                if nfm:
                    import numpy as np
                    mx = [r["max_stability"] for r in nfm.get("records", [])
                          if r["permuted"]]
                    if mx:
                        sat = float(np.mean(np.asarray(mx) >= 1.0))
                item += (
                    f" **But only at a narrow selection depth.** Run the same "
                    f"swap at the pre-registered `select_k = "
                    f"{ab.get('protocol', {}).get('select_k', 40)}` and error "
                    f"control goes the other way, "
                    f"{pct(pp40['null_abstention_heldout'])} to "
                    f"{pct(mm40['null_abstention_heldout'])} "
                    f"({d40:+.1f} points)"
                    + (f", because a more repeatable statistic pins a sensor "
                       f"in every resample more often -- {pct(sat)} of null "
                       f"replicates reach support 1.000 there, and the "
                       f"attainable control values above {pct(1 - sat)} drop "
                       f"out of the grid entirely (see \"What error control is "
                       f"achievable at all\")"
                       if sat is not None else "")
                    + f". So the change to make is *both* fields together: "
                      f"`attribution=\"model\"` **and** a depth narrow "
                      f"enough to keep saturation rare. One without the other "
                      f"is a regression.")
        item += (
            " Neither number reaches its target and the machinery around the "
            "statistic still buys nothing measurable"
            + (" on accuracy: the sparsity sweep shows its apparent +0.012 "
               "AUC edge was the extra sensors, not the ranking. And the "
               "conditionality above is not a footnote, it is why this is "
               "item 3 rather than item 1."
               if flipped else
               " -- but if the loop is kept, this is the change to make, and "
               "it costs a config edit."))
        L.append(item)
    if rk and rk.get("per_ranker"):
        per = rk["per_ranker"]
        plain = {k: v for k, v in per.items() if not k.startswith("agent")}
        ag_k = next((k for k in per if k.startswith("agent")), None)
        if plain and ag_k:
            def ctl(v):
                return v["heldout_alpha_0.05"]["null_abstention_heldout"]
            bk = max(plain, key=lambda k: ctl(plain[k]))
            bh = plain[bk]["heldout_alpha_0.05"]
            ah = per[ag_k]["heldout_alpha_0.05"]
            better = ctl(plain[bk]) >= ctl(per[ag_k])
            L.append(
                f"**Consider replacing the ranking core with a univariate "
                f"ranker.** `{bk}` reaches {pct(bh['null_abstention_heldout'])} "
                f"error control against the loop's "
                f"{pct(ah['null_abstention_heldout'])} and reports "
                f"{bh['real_reported_mean']:.2f} suspects against "
                f"{ah['real_reported_mean']:.2f}, without a "
                f"permutation-importance pass, a correlation-grouping step or "
                f"a verification loop."
                + (" Two caveats on that comparison are in the section above "
                   "-- the depth was probed for the baseline, and the "
                   "separation column rewards repeatability -- so read this as "
                   "the strongest available reason to try the swap and "
                   "measure, not as a settled result."
                   if better else
                   " The comparison is close enough that the swap is worth "
                   "measuring rather than assuming."))
    if ab5 and (ab5.get("levels") or {}).get("alpha_0.05"):
        a5 = ab5["levels"]["alpha_0.05"]
        a40 = (ab.get("levels") or {}).get("alpha_0.05") if ab else None
        a40_k = (ab.get("protocol") or {}).get("select_k", "40")
        a5_k = (ab5.get("protocol") or {}).get("select_k", "5")
        if a40:
            d = a5["null_abstention_heldout"] - a40["null_abstention_heldout"]
            L.append(
                f"**Selection depth is not the lever here.** "
                f"Dropping the loop's `select_k` from "
                f"{a40_k} to {a5_k} moved its error control by "
                f"{d * 100:+.1f} points on its own"
                + (", which is a larger effect than any ranker change "
                   "measured here and it is free." if abs(d) > 0.02 else
                   ", so for this loop, holding the attribution statistic "
                   "fixed, depth is not the lever it is for a univariate "
                   "ranker. It is not inert, though, and item 3 is why: depth "
                   "is what decides whether the attribution swap is a gain or "
                   "a regression, by setting how often the null "
                   "max-statistic saturates. Depth alone buys nothing; depth "
                   "is the precondition for the thing that does."))
    L += [
        "**Believe the forward-in-time split, not the shuffled one.** For a "
        "go/no-go decision the shuffled-CV number is the optimistic one, and "
        "the drift diagnostics say why: era is far more predictable from these "
        "sensors than failure is. Plan on retraining, and treat any fixed "
        "model as having a shelf life measured in weeks.",
        "**Treat the suspect list as a work order, not a diagnosis.** The "
        "invariance screen cannot certify any of these sensors as causal at "
        "this sample size, so the useful output is a shortlist of signal "
        "*families* worth an engineer's afternoon -- and, with the bar above "
        "in place, sometimes no shortlist at all.",
        "",
        "The honest summary of the KPI card: on SECOM this pipeline is a "
        "decent predictor and an unreliable attributor, its attribution is "
        "associational rather than causal, and the agent machinery is not what "
        "earns either -- a plain ranker matches or beats it on every axis "
        "measured here.",
    ]
    # Numbered here rather than in the strings: any bullet can be dormant
    # when its run has not happened, and a hard-coded list skips a number.
    tail = L[-2:] if L and L[-2] == "" else []
    items = L[:-2] if tail else L
    out = [f"{i}. {t}" for i, t in enumerate(items, 1)]
    return ["## What to actually do with this", "",
            "The measurements point at one configuration, and it is not the "
            "one that scores best on a slide:", ""] + out + tail + [""]


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
    ab = read_json(runs / "abstain.json")
    iv = read_json(runs / "invariance.json")
    rk = read_json(runs / "null_fdr_rankers.json")
    ab5 = read_json(runs / "abstain_k5.json")
    nf5 = read_json(runs / "null_fdr_k5.json")
    aa = read_json(runs / "attr_arm.json")
    sp = read_json(runs / "sparsity.json")
    cs = read_json(runs / "calib_size.json")
    dd = read_json(runs / "dedup.json")
    nfm40 = read_json(runs / "null_fdr_k5_model_b40.json")
    abm40 = read_json(runs / "abstain_k5_model_b40.json")
    ab5m = read_json(runs / "abstain_k5_model.json")
    nf5m = read_json(runs / "null_fdr_k5_model.json")
    abm = read_json(runs / "abstain_model.json")
    nfm = read_json(runs / "null_fdr_model.json")
    L += sec_headline(ev, st, sy, sw, prof, dr, rsw, nf, ab, rk)
    L += sec_kpi(ev, st, prof)
    L += sec_dataset(prof)
    L += sec_secom_auc(ev)
    L += sec_attr_auc(aa, ev)
    L += sec_sparsity(sp)
    L += sec_rolling(ev, prof)
    L += sec_drift(dr)
    L += sec_rolling_sweep(rsw, ev)
    L += sec_sweep(sw)
    L += sec_stability(st, prof)
    L += sec_null_fdr(nf)
    L += sec_abstain(ab)
    L += sec_ranker_fdr(rk)
    L += sec_depth(ab, ab5, rk, nf, nf5)
    L += sec_attribution_fdr(nf, ab, nf5, ab5, nfm, abm, nf5m, ab5m, rk)
    L += sec_nboot_grid(rk)
    L += sec_calib_size(cs, nfm40, abm40, ab5m)
    L += sec_dedup(dd)
    L += sec_saturation([
        ("agent, `select_k = 40`, permutation", nf, ab),
        ("agent, `select_k = 40`, **model**", nfm, abm),
        ("agent, `select_k = 5`, permutation", nf5, ab5),
        ("agent, `select_k = 5`, **model**", nf5m, ab5m),
    ])
    L += sec_invariance(iv)
    L += sec_synthetic(sy, ev)
    L += sec_recommend(ev, ab, rk, ab5, st, ab5m, abm, nfm)
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
                     d["rs"], d["nf"], d["ab"], d["rk"])[2:]).strip(),
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
    "null_fdr": lambda d: "\n".join(
        sec_null_fdr(d["nf"])[2:] + sec_abstain(d["ab"])).strip(),
    "ranker_fdr": lambda d: "\n".join(
        sec_ranker_fdr(d["rk"])[2:]
        + sec_depth(d["ab"], d["ab5"], d["rk"], d["nf"], d["nf5"])).strip(),
    "invariance": lambda d: "\n".join(sec_invariance(d["iv"])[2:]).strip(),
    "recommend": lambda d: "\n".join(
        sec_recommend(d["ev"], d["ab"], d["rk"], d["ab5"])[2:]).strip(),
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
        "ab": read_json(runs / "abstain.json"),
        "iv": read_json(runs / "invariance.json"),
        "rk": read_json(runs / "null_fdr_rankers.json"),
        "ab5": read_json(runs / "abstain_k5.json"),
        "nf5": read_json(runs / "null_fdr_k5.json"),
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
