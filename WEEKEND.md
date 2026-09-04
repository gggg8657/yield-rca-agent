# WEEKEND.md — yield-rca-agent, S2

*Written for a tired person. Five minutes. Long form is in
[`critique_log.md`](critique_log.md) and [`paper_draft.md`](paper_draft.md);
every number below is regenerated into [`RESULTS.md`](RESULTS.md) by
`scripts/report.py` from a JSON in `runs/`, and `scripts/report.py --check`
fails CI if a document and a JSON disagree.*

Last updated: Friday 2026-09-04, ~15:00.

---

## The one-paragraph version

Friday morning's session established the accuracy story: on real SECOM a plain
random forest beats the multi-agent loop, and the top-5 stability KPI is missed
by a wide margin. This session added the axis that was missing, and it is worse
news for the architecture. **The loop reports root causes on data where none
exist — 2,743 of them across 200 label-permuted replicates, abstaining on
exactly none**, so the safeguard the project is pitched on does not hold. I
implemented a null-calibrated abstention rule that fixes it and measured what it
costs. Then I checked whether a plain univariate ranker would do the same job:
it does it better, on every property measured. **The agent loop now has no
measured advantage over a univariate ranker on any axis in this repository** —
accuracy, stability, signal separation, or calibratable error control. It wins
only on the synthetic generator, where its premise is true by construction.

---

## Headline numbers: Friday morning → now

| | Friday morning | now | source |
|---|---|---|---|
| SECOM AUC, best plain baseline (`rf_all`) | 0.759 [0.739, 0.779] | unchanged | `runs/secom_eval.json` |
| SECOM AUC, agent loop (`agent_rf`) | 0.717 [0.699, 0.735] | unchanged | same |
| agent − baseline, paired over 25 folds | −0.042 [−0.058, −0.025] | unchanged | same |
| top-5 stability, agent loop, 200 bootstraps | 22.3% | unchanged | `runs/secom_stability.json` |
| **false-discovery rate under a no-cause null** | *[not measured]* | **1.00** | `runs/null_fdr.json` **(new)** |
| **abstention rate on that null** | *[not measured]* | **0.0%** | same **(new)** |
| **suspects SECOM supports at α = 0.05** | *[not measured]* | **0.60**, empty 51% of the time | `runs/abstain.json` **(new)** |
| **sensors associated with failure at all** | *[not measured]* | **22 of 474** | `runs/invariance.json` **(new)** |
| **those non-invariant across production periods** | *[not measured]* | **1** — and it is the loop's top suspect | same **(new)** |
| **error control the loop's confidence can reach** | *[not measured]* | **91.6%** (95% target) | `runs/null_fdr_rankers.json` **(new)** |
| **the same, for a plain univariate ranker** | *[not measured]* | **94.3%**, reporting 2.06 suspects vs the loop's 0.60 | same **(new)** |

Nothing from Friday morning was re-run or revised. The two KPI verdicts stand:
**AUC ≥ 0.75 met by the baseline, not by the agent loop; top-5 stability ≥ 80%
not met on any protocol.**

---

## The four new results

### 1. The loop invents root causes, and not for the reason the code suggests

Permute the labels (104 fails / 1,463 passes preserved exactly, `X` untouched),
run the whole loop, and every sensor it names is false by construction. Over 200
such replicates it named **2,743** and abstained **0** times.

I predicted the cause was the two `if not surv: surv = reps[:5]` guards in
`estimator.py`. **They fired on 0.0% of replicates.** The real cause is that
`stability_min = 0.3` is far too low: a sensor need only reach the top 40 of a
60-sensor pool in 4 of 12 bootstrap replicates, which noise does routinely, and
13.7 noise sensors clear it unaided.

I also predicted the null and real support distributions would overlap. They do
not — P(real > null) = **0.873**, Mann-Whitney p = 1.6e-14. Being wrong there is
the difference between a repairable pipeline and a dead one.

The obvious counter-explanation is refuted: if the loop were merely re-reporting
SECOM's correlation structure, null replicates would agree with each other. They
agree on **0.014** of their top-5 against a **0.011** random floor, having named
417 of 474 sensors at least once. The invented causes are fresh every time.

### 2. A fix that works, with its price measured

`AgentRCA(report_tau=...)` reports only suspects clearing τ(α), the (1−α)
quantile of the null's maximum support (Westfall–Young max-statistic,
family-wise, no independence assumption — which matters when 179 SECOM sensors
have a partner correlated above |r| = 0.99). It governs `reported_` only, so
`selected_` and `predict_proba` are unchanged and **enabling it cannot move any
AUC in this repo** (asserted in tests).

Priced with held-out calibration (τ fitted on half the null replicates, rates
read off the other half, 400 random partitions, zero refits):

| α | τ | null correctly silent | SECOM reports nothing | SECOM suspects |
|---|---|---|---|---|
| 0.10 | 0.842 | 82.0% (target 90%) | 23.4% | 1.18 |
| 0.05 | 0.910 | 91.6% (target 95%) | 51.3% | 0.60 |
| none | — | 0.0% | 0.0% | 20.95 |

Two shortfalls, both reported rather than smoothed: it **under-abstains** (91.6%
vs nominal 95%, because a point estimate of an upper quantile is biased low),
and τ sits on a **13-point grid** because `n_boot = 12`.

### 3. The suspects are associational, and the top one fails an invariance test

22 of 474 sensors are associated with failure (BH FDR 0.05, exact permutation,
B = 20,000). Asking which survive a change of production period — the marginal
Invariant-Causal-Prediction screen, with the null permuting *block membership*
so each sensor's overall association is held fixed — exactly **one** is
rejected: `sensor_059`, which the loop puts in its top-5 in **25 of 25** CV
folds. Per-period AUCs 0.56 / 0.77 / 0.86 / 0.52 / 0.49, I² = 0.82, p = 0.012.

**The other 21 were not testable, not invariant.** Injected sensors with a known
break are detected 14% / 23% / 52% / 79% / 95% of the time at first-period AUCs
of 0.55–0.75, while SECOM's associated sensors carry |AUC − 0.5| of only
0.088–0.192. Their entire signal is smaller than the break the screen needs. So
the honest statement is that **the pipeline reports associational suspects** and
this dataset cannot upgrade that.

### 4. A univariate ranker does the whole job better

The obvious question after result 2 is whether anything simpler would have
calibrated as well. Matched to the loop's own bootstrap count and selection
depth, plain rankers separate real from permuted labels *better* than the loop
(`univariate` 0.943 vs 0.873) but their support **saturates** at 1.000, which
caps their error control at 88.5% — below the loop's 91.6%. For one turn that
looked like the architecture's first genuine win, and it was written up as such.

It was wrong. What saturated was the *selection depth*, not the ranker: "top 40
of 474 in all 12 resamples" is easy, "top 5 of 474 in all 40" is not.

| arm | separation | error control | ceiling | suspects | abstains |
|---|---|---|---|---|---|
| `univariate` B=40, k=5 | **1.000** | **94.3%** | 100% | **2.06** | 0% |
| `univariate` matched (k=40) | 0.943 | 88.5% | 88.5% | 4.10 | 0% |
| **agent (full loop)** | 0.873 | 91.6% | 98.5% | 0.60 | 51% |

The univariate variant takes every column, with no permutation-importance pass,
no correlation grouping and no verification loop. **So the loop has no measured
advantage on any axis here.** `RESULTS.md` carries an explicit note that this
replaces the earlier conclusion rather than quietly overwriting it.

---

## What I tried that did not work, and what it rules out

- **Blaming the never-empty fallback for the false discoveries.** Measured at
  0.0%. Rules out "delete four lines and it's fixed" — the threshold is the
  problem, so the fix has to be a calibrated bar.
- **Describing a protocol the code did not implement.** An adversarial review
  caught `null_fdr_rankers.py`'s docstring and the generated `RESULTS.md` both
  claiming `SensorCleaner` is fitted inside each bootstrap resample. It is
  fitted once per replicate, above the loop. The numbers do not move — the
  cleaner is unsupervised, so a label permutation cannot change its output —
  but the description was wrong and is now corrected in three places. Rules out
  trusting a protocol string that no test reads.
- **Treating `P(real > null)` as a pure signal-detection measure.** The same
  review pointed out it is confounded with *repeatability*: "real" replicates
  reuse one label vector and vary only bootstrap randomness, so a
  near-deterministic univariate ranker is favoured over the loop's much wider
  internal stochasticity. The error-control column does not share the confound,
  so the conclusion holds — but separation is now labelled the weaker column.
- **Concluding the loop was the only arm that could carry a false-discovery
  guarantee.** True at the matched operating point, false as a statement about
  the architecture, and I committed it before the follow-up run refuted it. Rules
  out reading any single operating point as a property of the design — and it is
  the reason every report section here is generated from JSON with data-driven
  branches, so a refuted sentence turns into a diff rather than surviving.
- **Permuting labels as the invariance null.** Reported 42 non-invariant sensors;
  the correct block-membership null reports 1. Permuting labels builds the
  reference at AUC 0.5, where the statistic has less sampling variance than at
  the observed association, so it convicts strong sensors for being strong. The
  wrong JSON was deleted rather than kept. Rules out the closed-form and the
  naive-null versions of this test.
- **The closed-form χ² reference for Cochran's Q.** Rejects at 0.061 under the
  null against a nominal 0.050 because SECOM's sensors are heavily tied. Rules
  out asymptotic p-values here; everything uses permutation.
- **`agy` and `cursor-agent` as second opinions.** `cursor-agent` needs
  credentials I do not have; `agy` needs a permission grant headless, and the
  suggested `--dangerously-skip-permissions` would let another agent write in
  this repo while my jobs ran in it. Declined deliberately. `codex` worked and
  independently found the fallback flaw — quoted in `critique_log.md`.

---

## Decisions that need a human

### Decision 1 — What the KPI card should say

`codex`, reviewing adversarially, made a point I could not dismiss: the AUC KPI
is scored against **shuffled** CV, whose folds mix production eras, and the
sensors identify era at adversarial AUC 0.993. The same baseline falls to 0.532
on a chronological split. The card's 0.759 also has a CI [0.739, 0.779] that
crosses the 0.75 line.

- **(a) Leave it.** Card reads "met", protocol stated, forward-in-time numbers
  reported adjacently. Defensible and already true, but a reader who skims sees
  a met KPI backed by the optimistic protocol.
- **(b) Score the KPI forward-in-time.** Honest for deployment; the KPI then
  fails for every arm, and it is a change of definition after seeing results.
- **(c) Report both verdicts side by side** as "met (shuffled CV) / not met
  (forward in time)".

**My recommendation: (c).** It changes no number and removes the only reading of
the card that is misleading.

### Decision 2 — Whether to keep the agent loop at all

This is the decision the weekend's results actually force, and it was not on
Friday morning's list.

- **(a) Keep it as the product.** Defensible only if the deliverable is the
  written report and the correlation grouping, neither of which any measurement
  here scores. Nothing measured supports it on accuracy, stability, separation
  or error control.
- **(b) Keep it, retargeted at the regime where it wins.** The synthetic
  contrast is sharp and reproducible: where a few sensors genuinely dominate,
  the loop beats the full-sensor forest and *meets* the stability KPI. Ship it
  with a documented precondition and a cheap pre-flight test for signal
  concentration.
- **(c) Replace the ranking core with a univariate ranker at a non-saturating
  depth**, keep the agent framing for reporting and grouping, and keep the
  null-calibrated abstention rule. This is what the numbers point at.

**My recommendation: (c), with (b)'s precondition documented.** It keeps
everything that measures well and drops the part that does not. It is a large
change to the project's premise, which is exactly why it is a decision and not
something I made unilaterally.

### Decision 3 — Whether to ship abstention on by default

`report_tau` is implemented and off by default, so nothing has changed yet.

- **(a) Off by default.** Preserves every existing number; the FDR result stays a
  documented flaw rather than a fixed one.
- **(b) On at α = 0.10.** SECOM reports ~1.2 suspects, empty 23% of the time.
- **(c) On at α = 0.05.** ~0.6 suspects, empty 51% of the time.

**My recommendation: (b).** It restores meaningful control while still usually
returning something an engineer can act on, and α is a business call about the
cost of a wasted investigation versus a missed cause — which is why I did not
make it unilaterally. Note that under (b) or (c) the deliverable changes shape:
the product is no longer "your five root causes" but "at most one or two, often
none, and the honest reason why".

### Decision 4 — Whether the stability KPI is attainable on SECOM at all

With 104 fails, 70% of sensors non-stationary, and only 22 sensors showing any
association, ≥80% top-5 agreement may not be reachable by any method on this
dataset. Options: keep it and report the miss; redefine it at cluster level
(best ranker reaches 73.7% under CV-fold perturbation); or move the KPI to a
dataset with more failures. This was Friday morning's open question too and it
is still open — it needs a call, not more measurement.

---

## Still running / how to check

| job | started | expect | check |
|---|---|---|---|
| `scripts/null_fdr.py --select-k 5` (H4: is depth what limits the loop's error control too?) | ~13:25 Fri | ~33 min, writes `runs/null_fdr_k5.json` | `tail -5 runs/null_fdr_k5.log` |

Queued behind it, in order:

1. **H5** — `scripts/stability_secom.py --only agent_model --append`. Re-measures
   the headline top-5 stability KPI with `attribution="model"` instead of
   held-out permutation importance, one config field changed. Three separate
   measurements now implicate that statistic; this tests whether it is what
   limits stability, or whether the sample-size wall dominates as this repo
   currently claims. ~80 min. Hypothesis and both competing predictions are in
   `critique_log.md` Turn 8, written before the run.
2. **Re-run `scripts/null_fdr_rankers.py --variants`** so its JSON's own
   `leakage_control` field carries the corrected protocol text. The numbers are
   unaffected; `RESULTS.md` already states the corrected version from a
   verified literal. ~32 min, cosmetic.

H4 asks *why* the loop loses, not whether — the univariate arm has already
cleared it either way. If the loop's control climbs into the 93–94% band when
only its selection depth changes, the finding narrows to "selection depth
dominates ranker choice", which is more useful to a practitioner than "this
architecture underperforms". If it does not move, its permutation-importance
estimator is the binding constraint. Hypothesis written down before the run
(`critique_log.md`, Turn 6).

Reproduce everything: `bash scripts/overnight.sh ~/miniforge3/envs/pybamm-inv/bin/python`
(11 stages, CPU only, 16 workers). Tests: `tests/test_smoke.py` (2),
`tests/test_real.py` (19), `tests/test_null.py` (11) — all green as of the last
commit, which is pushed.
