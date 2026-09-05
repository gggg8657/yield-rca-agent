# WEEKEND.md — yield-rca-agent, S2

*Five minutes. Long form: [`critique_log.md`](critique_log.md) (every
hypothesis, verdict and retraction, in order) and
[`RESULTS.md`](RESULTS.md) (every number, generated from `runs/*.json`).
`scripts/report.py --check` fails CI if a document and a JSON disagree;
`scripts/audit_weekend.py` fails if any number in **this** file does not trace
to a run.*

Last updated: Saturday 2026-09-05, ~08:40.

---

## The one-paragraph version

Friday morning's session established the accuracy story: on real SECOM a plain
random forest beats the multi-agent loop and the top-5 stability KPI is missed
badly. This session added the axis that was missing and then spent fifteen
experiments trying to save the architecture. **The loop reports root causes on
data where none exist** — false-discovery rate 1.00 under a label-permuted null
— so the safeguard the project is pitched on does not hold; I implemented a
null-calibrated abstention rule that fixes it and priced it. Then I tuned three
of the loop's parameters in its own favour, one at a time, each pre-registered.
The best configuration found is **level with a plain univariate ranker on error
control and behind it on every other axis.** The architecture's one measured
win is that its correlation grouping really does remove a near-duplicate pair
the univariate ranker always leaves in. That is the honest summary: a decent
predictor, an unreliable attributor, and machinery that is not what earns
either.

---

## Headline numbers: Friday morning → now

| | Friday morning | now | source |
|---|---|---|---|
| SECOM AUC, best plain baseline (`rf_all`) | 0.759 [0.739, 0.779] | unchanged | `secom_eval.json` |
| SECOM AUC, agent loop (`agent_rf`) | 0.717 [0.699, 0.735] | unchanged | same |
| agent − baseline, paired over 25 folds | −0.042 [−0.058, −0.025] | unchanged | same |
| SECOM AUC, loop after tuning its attribution | *[not measured]* | 0.729 [0.710, 0.748], still −0.0303 [−0.0474, −0.0132] behind | `attr_arm.json` |
| top-5 stability, agent loop | 22.3% | 35.3% after tuning attribution | `secom_stability.json` |
| **false-discovery rate under a no-cause null** | *[not measured]* | **1.00** | `null_fdr.json` |
| **error control the loop can reach**, after tuning 3 parameters | *[not measured]* | **94.2%** on 1.55 suspects | `abstain_k5_model_b40.json` |
| **the same, for a plain univariate ranker** | *[not measured]* | **94.3%** on 2.06 suspects | `null_fdr_rankers.json` |
| **sensors associated with failure at all** | *[not measured]* | **22 of 474** | `invariance.json` |
| **those non-invariant across production periods** | *[not measured]* | **1** — and it is the loop's top suspect | same |

**Both KPI verdicts stand: AUC ≥ 0.75 met by the baseline, not by the agent
loop; top-5 stability ≥ 80% not met on any protocol or configuration.**

---

## Decisions that need a human

### Decision 1 — Whether to keep the agent loop at all

The decision the weekend's results force, and it was not on Friday's list.

- **(a) Keep it as the product.** Nothing measured supports it on accuracy,
  stability, separation or error control.
- **(b) Keep it, retargeted at the regime where it wins.** The synthetic
  contrast is sharp: where a few sensors genuinely dominate, the loop beats the
  full-sensor forest and *meets* the stability KPI. Ship with a documented
  precondition and a cheap pre-flight test for signal concentration.
- **(c) Replace the ranking core with a univariate ranker at a non-saturating
  depth**, keep the agent framing for reporting and grouping, keep the
  null-calibrated abstention rule.

**Recommendation: (c), with (b)'s precondition documented.** Keeps everything
that measures well, drops what does not. It is a large change to the project's
premise, which is why it is a decision and not something I made unilaterally.

### Decision 2 — Whether to ship abstention on by default

`report_tau` is implemented and **off** by default, so nothing has changed yet.

- **(a) Off.** Preserves every existing number; the FDR result stays a
  documented flaw rather than a fixed one.
- **(b) On at α = 0.10.** Roughly one to two suspects, sometimes empty.
- **(c) On at α = 0.05.** Fewer suspects, empty far more often.

**Recommendation: (b).** α is a business call about the cost of a wasted
investigation versus a missed cause, which is why I did not make it. Note that
under (b) or (c) the deliverable changes shape: not "your five root causes" but
"at most one or two, often none, and the honest reason why".

### Decision 3 — What the KPI card should say

The AUC KPI is scored against **shuffled** CV, whose folds mix production eras;
the same baseline falls to 0.532 on a chronological split, and the 0.759 CI
crosses the 0.75 line.

- **(a) Leave it** — defensible, but a skimming reader sees a met KPI backed by
  the optimistic protocol.
- **(b) Score it forward-in-time** — honest for deployment, but the KPI then
  fails for every arm and it is a definition change made after seeing results.
- **(c) Report both verdicts** as "met (shuffled CV) / not met (forward in
  time)".

**Recommendation: (c).** Changes no number, removes the only misleading reading.

### Decision 4 — Whether the stability KPI is attainable on SECOM at all

With 104 fails and only 22 sensors showing any association, ≥80% top-5
agreement may not be reachable by any method here. Keep it and report the miss;
redefine it at cluster level; or move it to a dataset with more failures. Open
since Friday — it needs a call, not more measurement.

---

## What changed, in one line each

Detail for every row is in `RESULTS.md`; the reasoning, including what I
predicted beforehand and got wrong, is in `critique_log.md`.

| # | finding | verdict |
|---|---|---|
| 1 | The loop reports root causes on permuted labels; FDR is 1.00 and it never abstains | flaw confirmed |
| 2 | A null-calibrated bar (`report_tau`) fixes it; cost measured in suspects and abstention | fix works |
| 3 | Suspects are *associational*; the one sensor an invariance screen can adjudicate is the loop's favourite, and it fails | negative |
| 4 | A univariate ranker matches or beats the loop on accuracy, stability, separation and error control | negative |
| 5 | Selection depth is not the loop's constraint — moves its error control by −0.0 points | refuted |
| 6 | Swapping the attribution statistic is worth **+13.0** stability points; the architecture around it about **+2** | localised |
| 7 | Same swap on error control: **+2.1** points at a narrow depth… | H6 confirmed |
| 8 | …and **−3.6** at the pre-registered depth, because a more repeatable statistic saturates the null max-statistic | H6 reversed |
| 9 | Error control lives on a grid of spacing 1/`n_boot`; nominal 0.95 is simply not in the set at `n_boot` = 12 | mechanism |
| 10 | Raising `n_boot` to 40 refines the grid and reaches nominal; `P(M = 1)` *fell*, refuting the competing story | H9 confirmed |
| 11 | The gap to nominal splits into a **grid** budget and a **calibration** budget — different spends, previously conflated | new |
| 12 | The loop's AUC deficit is **sparsity price in full**; the attribution statistic buys zero AUC | H8 refuted |
| 13 | `PredictAllReportFew` asserted an equality it had not earned; measuring it moved the number by −0.0021 | unearned, correct |
| 14 | The correlation grouping does what it advertises: the loop's top-5 never contains a near-duplicate pair, univariate's always does | H10, first win |
| 15 | Three parameters tuned in the loop's favour; on no axis does it finish ahead of ranking each sensor alone | headline |

---

## What did not work, and what it rules out

- **Blaming the never-empty fallback for the false discoveries.** It provably
  never reaches the calibrated report — 0 of 800 splits. Rules out "delete four
  lines and it's fixed"; the fix has to be a calibrated bar.
- **Three times, generalising a mechanism from one operating point.** Ranker
  saturation, the never-empty guard, and the attribution swap each held at one
  selection depth and reversed at another. Every mechanism claim in `RESULTS.md`
  now names the depth it was measured at.
- **Describing a protocol the code did not implement.** An adversarial review
  caught a docstring and the generated `RESULTS.md` both claiming `SensorCleaner`
  is fitted inside each bootstrap resample; it is fitted once per replicate. The
  numbers do not move — it is unsupervised — but rules out trusting a protocol
  string no test reads.
- **Calling the saturation ceiling an identity that predicts.** It is an
  inequality, one line from boundedness plus a `>=` comparison. What replaced it
  — the discreteness of the attainable set — is not vacuous and is actionable.
- **Treating `P(real > null)` as pure signal detection.** Confounded with
  repeatability, so it is labelled the weaker column and error control carries
  the argument.
- **Permuting labels as the invariance null.** Convicts strong sensors for being
  strong. The correct block-membership null reports 1 non-invariant sensor. The
  wrong JSON was deleted rather than kept.
- **The closed-form χ² reference for Cochran's Q.** Anticonservative on tied
  data. Everything uses permutation p-values.
- **`agy` and `cursor-agent` as second opinions.** `cursor-agent` needs
  credentials I do not have; `agy` headless wanted a permission grant that would
  let another agent write in this repo while my jobs ran. Declined deliberately.
  `codex` worked and found four real flaws, all quoted and fixed.

---

## Still running / how to check

```bash
scripts/jobs.sh    # one line per job; marks a dead one STALE
```

| job | expect | why |
|---|---|---|
| **H11** — `null_fdr.py` + `null_fdr_rankers.py`, re-run with a new `support_by_sensor` field | ~1 h | H10 could only measure the *ranking*; the τ-thresholded report was not reconstructible from disk. This records per-sensor support so the deduplication question can be asked of the reports themselves. Prediction registered in `critique_log.md` Turn 15. |

**A warning about liveness checks.** I lost ~5 hours on Friday night to a launch
that died silently: the tail of its log read exactly like a healthy just-started
run. `scripts/runjob.sh` now stamps a heartbeat and `scripts/jobs.sh` marks a
stale job dead. Use those, not `tail`.

Reproduce everything: `bash scripts/overnight.sh ~/miniforge3/envs/pybamm-inv/bin/python`
(CPU only, 16 workers). Tests: **48 collected, all green**.
