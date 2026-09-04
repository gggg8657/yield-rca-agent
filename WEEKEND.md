# WEEKEND.md — yield-rca-agent, S2

*Written for a tired person. Five minutes. Long form is in
[`critique_log.md`](critique_log.md) and [`paper_draft.md`](paper_draft.md);
every number below is regenerated into [`RESULTS.md`](RESULTS.md) by
`scripts/report.py` from a JSON in `runs/`, and `scripts/report.py --check`
fails CI if a document and a JSON disagree.*

Last updated: Friday 2026-09-04, ~12:00.

---

## The one-paragraph version

Friday morning's session established the accuracy story: on real SECOM a plain
random forest beats the multi-agent loop, and the top-5 stability KPI is missed
by a wide margin. This session added the axis that was missing and it is worse
news for the architecture. **The loop reports root causes on data where none
exist — 2,743 of them across 200 label-permuted replicates, abstaining on
exactly none.** The safeguard the project is pitched on does not hold. The good
news is that the underlying statistic is salvageable, so I implemented and
priced a fix; the bad news is that a plain univariate ranker may be *better*
calibrated than the loop, which is being measured right now.

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

Nothing from Friday morning was re-run or revised. The two KPI verdicts stand:
**AUC ≥ 0.75 met by the baseline, not by the agent loop; top-5 stability ≥ 80%
not met on any protocol.**

---

## The three new results

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

---

## What I tried that did not work, and what it rules out

- **Blaming the never-empty fallback for the false discoveries.** Measured at
  0.0%. Rules out "delete four lines and it's fixed" — the threshold is the
  problem, so the fix has to be a calibrated bar.
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

### Decision 2 — Whether to ship abstention on by default

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

### Decision 3 — Whether the stability KPI is attainable on SECOM at all

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
| `scripts/null_fdr_rankers.py` (H2: is a plain ranker better calibrated than the loop?) | ~11:55 Fri | ~20 min, writes `runs/null_fdr_rankers.json` | `tail -5 runs/null_fdr_rankers.log` |

At n = 3 (smoke) the plain rankers separated real from null at 1.000 against the
loop's 0.873 and supported 4–7 suspects at τ against the loop's 0.57. If that
holds at n = 200, the loop has now lost on accuracy, on stability **and** on
false-discovery calibration — the axis its verification machinery exists for —
and that becomes the headline. The hypothesis was written down before the run
(`critique_log.md`, Turn 4).

Reproduce everything: `bash scripts/overnight.sh ~/miniforge3/envs/pybamm-inv/bin/python`
(10 stages, CPU only, 16 workers). Tests: `tests/test_smoke.py` (2),
`tests/test_real.py` (19), `tests/test_null.py` (11) — all green as of the last
commit, which is pushed.
