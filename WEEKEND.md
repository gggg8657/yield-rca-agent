# WEEKEND.md — yield-rca-agent, S2

*Written for a tired person. Five minutes. Long form is in
[`critique_log.md`](critique_log.md) and [`paper_draft.md`](paper_draft.md);
every number below is regenerated into [`RESULTS.md`](RESULTS.md) by
`scripts/report.py` from a JSON in `runs/`, and `scripts/report.py --check`
fails CI if a document and a JSON disagree.*

Last updated: Friday 2026-09-04, ~21:40.

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
| **top-5 stability, loop with model-native attribution** | *[not measured]* | **35.3%** — +13.0 points for one config field | `runs/secom_stability.json` **(new)** |
| **error control, loop with model-native attribution** (`select_k = 5`) | *[not measured]* | **93.7%**, reporting 1.34 suspects | `runs/abstain_k5_model.json` **(new)** |

Nothing from Friday morning was re-run or revised. The two KPI verdicts stand:
**AUC ≥ 0.75 met by the baseline, not by the agent loop; top-5 stability ≥ 80%
not met on any protocol.**

---

## The seven new results

### 1. The loop invents root causes, and not for the reason the code suggests

Permute the labels (104 fails / 1,463 passes preserved exactly, `X` untouched),
run the whole loop, and every sensor it names is false by construction. Over 200
such replicates it named **2,743** and abstained **0** times.

I predicted the cause was the two `if not surv: surv = reps[:5]` guards in
`estimator.py`. **At the pre-registered `select_k = 40` they fired on 0.0% of
replicates.** The cause there is that `stability_min = 0.3` is far too low: a
sensor need only reach the top 40 of a 60-sensor pool in 4 of 12 bootstrap
replicates, which noise does routinely, and 13.7 noise sensors clear it unaided.
(Result 5 shows this diagnosis is depth-specific: narrow the depth and the guard
becomes the whole story.)

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

### 5. Why it loses is now localised: the depth is not the constraint

Running the loop at the univariate variant's selection depth (`select_k = 5`,
the only change) was the symmetric comparison the tuned table above was missing.

| at α = 0.05 | agent `k=40` | agent `k=5` | `univariate` `k=5` |
|---|---|---|---|
| no-cause worlds kept silent | 91.6% | **91.5%** | **94.3%** |
| suspects reported | 0.60 | 1.16 | 2.06 |
| reports nothing (real) | 51% | 6% | 0% |

**Depth does not move the loop's error control at all** (less than half a
point), while it moved the univariate arm substantially. So the binding
constraint is the held-out permutation-importance estimator, not the depth — a
property of the architecture rather than a parameter. Depth still buys a usable
report (0.60 → 1.16 suspects), it just does not close the gap. And the
equal-effort comparison lands where the tuned one did, so that conclusion was
not a tuning artifact.

The unpredicted part, on permuted labels:

| | `select_k = 40` | `select_k = 5` |
|---|---|---|
| noise sensors clearing the threshold **on merit** | 13.7 | **0.47** |
| never-empty fallback fired | **0.0%** | **62.5%** |
| replicates reporting nothing | 0.0% | 0.0% |

At the pre-registered depth the threshold is too loose to filter anything and
the fallback is never needed. At the narrow depth the threshold works almost
perfectly and the fallback fires on 62.5% of null replicates.

**I first wrote that up as the fallback destroying an otherwise working filter,
and that was wrong — see result 6.** The fallback tops up `selected_`, the set
the classifier is fitted on, which cannot be empty; the error-control column is
computed from the *pre-drop* supports and never reads it. The last row above is
0% at both depths because these runs set `report_tau = None`, i.e. abstention is
switched off by configuration — that row cannot be anything else, and I should
not have read it as a finding.

### 6. The one config field that is worth 13 points — and the architecture that is worth about two

The estimator diagnosis in result 5 was reached by elimination, so I tested it
on the KPI the project is scored against. `attribution="model"` instead of
held-out permutation importance, one field, same 200 bootstrap replicates.

| ranker | top-5 stability | wall |
|---|---|---|
| `univariate` — rank each sensor on its own | 46.1% | 0.2 min |
| `rf_impurity` — the forest's own importance | 36.5% | 1.0 min |
| **`agent_model` — full loop, model-native attribution** | **35.3%** | 14.6 min |
| `agent` — full loop, held-out permutation (pre-registered) | 22.3% | 40.3 min |
| `perm_only` — the loop's attribution step alone | 20.0% | 7.9 min |

Read as a 2x2 rather than a ladder, this decomposes the loop:

|  | permutation attribution | model-native attribution | statistic is worth |
|---|---|---|---|
| **bare ranker** (attribution step only) | `perm_only` 20.0% | *[not measured]* | *[not measured]* |
| **full agent loop** | `agent` 22.3% | `agent_model` 35.3% | **+13.0** |
| **architecture is worth** | +2.3 | *[not measured]* | |

The clean cell is the bottom row — one config field, everything else identical:
**+13.0 points and a 2.8x speedup** (40.3 min → 14.6 min). Against that, the
whole architecture (screen, correlation grouping, bootstrap verify, drop) is
worth **+2.3** points over the permutation statistic, inside one standard
deviation of the replicate-to-replicate spread (15.2 points).

The model-native architecture delta is blank on purpose. I first filled it with
`rf_impurity` (36.5%) and got −1.1, and an adversarial review
(`codex`, quoted in `critique_log.md` Turn 10) correctly rejected that:
`rf_impurity` fits a 500-tree forest over every cleaned sensor with no screen,
so subtracting it from `agent_model` prices the architecture *plus* a tree count
*plus* a candidate universe. The matched arm — the loop's own attribution step
with the other statistic and nothing after it — is now implemented
(`model_only`, built from `AgentRCA._rank` so it cannot drift, with a test
asserting its permutation twin reproduces `perm_only` exactly) and **queued
behind H6**. Until it lands the cell is a blank, not an estimate.

The prediction was written down before the run (`critique_log.md`, Turn 8):
"materially above 22.3%, landing near `rf_impurity`'s 36.5%, and will not reach
80%." Measured 35.3%. **The KPI is still missed by 44.7 points**, and this was
never an attempt to reach it — it was an attempt to attribute the miss to a
component, which now succeeds.

One caveat on the target itself, which I should have said earlier: this repo has
no published SECOM figure for top-5 selection stability to compare against.
SECOM papers report classification AUC. The 80% is a project target from the
brief with no external provenance, so the only honest comparisons are the
internal ones in the table.

### 6b. And a retraction: the guard was never the problem

Checking result 5's write-up against the code it describes rather than
re-running anything, the fallback claim does not survive. Split the 200 null
replicates at `select_k = 5` by whether the guard fired:

| null replicates, `select_k = 5` | n | largest support reached | naming a suspect over τ = 0.417 |
|---|---|---|---|
| fallback fired (nothing cleared π = 0.3) | 125 | 0.250 | **0** |
| threshold cleared on merit | 75 | 0.583 | 25 |

The guard fires exactly when every support is below π = 0.3, and τ(0.05) = 0.417
sits *above* π — so **every replicate it fires on is already silent under the
calibrated rule.** Zero of the 125 name anything. The null worlds that get
through are entirely ones where the attribution estimator handed a pure-noise
sensor a genuinely high support.

An adversarial review pushed back on exactly the right spot: τ = 0.417 is *one*
threshold fitted on the whole null, whereas the 91.5% headline refits τ on one
half of the replicates and counts on the other, so the cross-tab did not prove
the claim for every split. It is now counted inside that protocol instead —
per split, how many evaluation-half replicates both had the guard fire *and*
named a suspect over that split's own τ:

| `select_k = 5` | smallest τ fitted | splits where the guard reached the report |
|---|---|---|
| α = 0.1 | 0.333 | **0 of 800** |
| α = 0.05 | 0.417 | **0 of 800** |
| α = 0.01 | 0.417 | **0 of 800** |

Zero everywhere, and not by luck: the smallest τ any split fits still sits above
`stability_min` = 0.3, and the guard fires only when every support is below it.
The objection turned a claim resting on one threshold into one resting on an
ordering that holds across all 800.

So the guards are not the mechanism behind the false-discovery rate at *either*
depth — the original result-1 diagnosis was right and my correction to it was
wrong — and the residual error-control failure is now attributed to the
estimator by measurement rather than by elimination. Fixed in the generator,
pinned by two new tests, and the full reasoning is in `critique_log.md` Turn 10.
No headline number changes.

### 7. The same one-field change, on the error-control axis (H6)

Result 5 pinned the loop's error-control constraint to its attribution
estimator **by elimination** — depth ruled out, then the guard ruled out (6b).
This repo has twice had an elimination argument fail when tested directly, so it
got tested directly. `attribution="model"`, nothing else, same 800 split-half
calibrations.

| at `select_k = 5` | permutation | **model-native** | target |
|---|---|---|---|
| control, α = 0.1 | 84.8% | **85.8%** | 90% |
| control, α = 0.05 | 91.5% | **93.7%** | 95% |
| control, α = 0.01 | 97.5% | **98.2%** | 99% |
| suspects reported, α = 0.1 | 1.33 | **1.70** | — |
| suspects reported, α = 0.05 | 1.16 | **1.34** | — |
| suspects reported, α = 0.01 | 0.90 | 0.90 | — |
| real-label abstention (α = 0.05) | 6.2% | **0.0%** | — |
| separation, same protocol | 0.982 | **0.994** | — |

**Predicted "above 93.0%" before the run. Measured 93.7%.** Note the α = 0.01
row, which is the one that does not flatter it: the two arms tie at 0.90
suspects there, so the report-length advantage is confined to the looser levels. The competing
explanation — control capped by bootstrap variance over ~65 failed wafers
regardless of the statistic — predicted no movement and is refuted.

The pre-registered distrust check passes too: I said a confirmation whose
suspect count collapsed below ~1.0 would mean it bought control by saying less.
It went the other way — the report gets **longer** (1.16 → 1.34) while control
improves and real abstention falls to zero. A report-less-to-control-more rule
cannot move both columns the right way at once.

**Against the univariate baseline** at matched depth, α = 0.05:

| arm | control | suspects | separation |
|---|---|---|---|
| `univariate (n_boot=40, select_k=5)` | 94.3% | 2.06 | 1.000 |
| agent loop, model-native, `select_k = 5` | 93.7% | 1.34 | 0.994 |
| agent loop, permutation, `select_k = 5` | 91.5% | 1.16 | 0.982 |

The control gap goes 2.8 points → 0.6 and the separation gap 0.018 → 0.006. The
report-length gap does not close. **And that table has a confound I have to
state rather than enjoy:** the univariate arm resamples 40 times, the loop 12,
and two of those three columns are statistics *of* the bootstrap distribution.
A 0.6-point gap is well inside what that difference could account for. Read it
as *close*, not as either arm ahead.

**So the headline narrows rather than reverses.** Not "the loop is beaten on
every axis" but: the loop is not measurably *better* than a univariate ranker on
any axis, and the version that comes closest costs a one-field config change the
project had never tried. What the architecture adds on top of that statistic is
still +2.3 points of stability — inside one sd — and nothing measurable on
accuracy.

---

## What I tried that did not work, and what it rules out

- **Blaming the never-empty fallback for the false discoveries.** Measured at
  0.0%. Rules out "delete four lines and it's fixed" — the threshold is the
  problem, so the fix has to be a calibrated bar.
- **Predicting that selection depth explains the loop's weak error control.**
  It explains the *baseline's* entirely and the loop's not at all (91.6% →
  91.5%). Rules out tuning `select_k` as a fix, and localises the problem in the
  permutation-importance estimator.
- **Twice now, generalising a mechanism from one operating point.** The ranker
  saturation held at one selection depth and vanished at another; the
  never-empty guard was irrelevant at one depth and decisive at another. Rules
  out single-operating-point mechanism claims on this pipeline; every such claim
  in `RESULTS.md` now names the depth it was measured at.
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
| **H6, second leg** — the same `--attribution model` run at the pre-registered depth | ~22:05 Fri | `runs/null_fdr_model.json` + `runs/abstain_model.json` | `tail -5 runs/null_fdr_model.log`; done when it ends with `H6 DONE` |

**The `select_k = 5` leg has landed and is result 7 above (93.7%, confirmed).**
The second leg runs the identical change at the pre-registered depth, because
the one lesson from Turn 9 that survived is that a single operating point is
never enough — and on this pipeline two mechanism claims have already failed to
generalise between exactly these two depths. So the honest expectation is that
it might not replicate, and if it does not, result 7's scope shrinks to
`select_k = 5` rather than the claim being retracted.

Nothing in the documents depends on it: `scripts/report.py` pairs arms by depth
and only emits the pre-registered comparison once both of its JSONs exist, so
the section will appear when the run lands and `--check` will flag `RESULTS.md`
as stale until it is regenerated.

| job | started | expect | check |
|---|---|---|---|
| **`model_only`** — the matched bare-ranker cell of the result-6 2x2 | queued behind H6 | fills the `*[not measured]*` blank in `runs/secom_stability.json` | `tail -3 runs/stability_model_only.log`; done when it ends with `MODEL_ONLY DONE` |

`model_only` is the loop's own attribution step with model-native importance and
nothing after it — the arm that makes the model-native column's architecture
delta a matched subtraction instead of a confounded one. It waits on H6's
completion marker rather than running alongside it, since both want the same 16
workers. Once it lands, `scripts/report.py` fills the blank automatically and
`--check` will flag `RESULTS.md` as stale until it is regenerated.

Also queued, cosmetic: re-run `scripts/null_fdr_rankers.py --variants` so its
JSON's own `leakage_control` field carries the corrected protocol text. The
numbers are unaffected and `RESULTS.md` already states the corrected version
from a verified literal. ~32 min.

Reproduce everything: `bash scripts/overnight.sh ~/miniforge3/envs/pybamm-inv/bin/python`
(11 stages, CPU only, 16 workers). Tests: **41 collected, all green** as of the
last commit (`tests/test_smoke.py` 6, `tests/test_real.py` 19,
`tests/test_null.py` 16). The per-file counts had drifted, which is why
`scripts/audit_weekend.py` now checks them too.
