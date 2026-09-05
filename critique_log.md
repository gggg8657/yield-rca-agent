# Critique log — S2, yield root-cause copilot

Running record of what was measured, what it was measured against, and where the
explanation could be wrong. Numbers cited here are read out of `runs/*.json` by
`scripts/report.py`; where a number appears in prose it is because a run in this
repository produced it and the JSON is named next to it.

Conventions used throughout:

* **synthetic** = the generator in `yieldrca.data.make_synthetic`, which has
  ground-truth causal sensors. **real** = UCI SECOM, which has none. A claim
  about recovering causes can only ever be synthetic.
* A metric nobody ran is written `[not measured]`.

---

## Turn 1 (2026-09-04) — audit before measurement

### What the previous session left

`DONE_OVERNIGHT.md` reports a complete SECOM evaluation. I re-ran
`scripts/report.py --check`: **"report is in sync with runs/*.json"**, so the
documents and the JSONs agree and I am not auditing stale prose.

Headline as inherited, all from `runs/secom_eval.json` and
`runs/secom_stability.json`:

| KPI | arm | value | target | verdict |
|---|---|---|---|---|
| SECOM AUC | `rf_all` (plain forest) | 0.759 [0.739, 0.779] | ≥0.75 | met |
| SECOM AUC | `agent_rf` (the agent loop) | 0.717 [0.699, 0.735] | ≥0.75 | **not met** |
| top-5 stability | `agent_rf`, 200 bootstraps | 22.3% | ≥80% | **not met** |

The agent loop loses to a plain random forest by −0.042 [−0.059, −0.025] over 25
paired folds. That is the finding, and the README already leads with it rather
than burying it — checked, `README.md` states it in the first results table.

### Leak audit (the brief asked for two specific culprits, explicitly)

Published SECOM AUCs cluster in 0.70–0.80, so 0.759 is inside the plausible band
rather than above it — which is weak evidence of no leak, not proof. I checked
the two named mechanisms directly:

1. **Timestamp used as a feature.** *Not present.* `yieldrca/data.py:50` builds
   `X` from `secom.data` alone; the timestamp lives in the *label* file and is
   only returned when a caller passes `with_time=True`. Every caller of that
   flag (`prepare_data.py:44`, `eval_secom.py:66`, `drift.py:66`,
   `rolling_sweep.py:64`) binds it to a separate `t` and uses it only to build
   splits. `grep` for `hstack`/`column_stack` across `scripts/` and `yieldrca/`
   returns three hits, none of which touch `t`
   (`preprocess.py:123` appends missing-indicators, `estimator.py:255,335`
   assemble a two-column probability). `X.shape == (1567, 590)` at load, i.e.
   exactly the sensor count with no extra column.

2. **Imputation or scaling fitted outside the fold.** *Not present.* Every
   imputer and scaler is a step inside a `Pipeline` (`arms.py:60-105`) or inside
   `AgentRCA.fit` (`estimator.py:191`), so `cross_validate` fits them on
   training rows only. The one documented exception is the correlation-cluster
   map used for the cluster-aware stability variant
   (`yieldrca/stability.py:53`), which is built from the **unlabelled** sensor
   matrix and never used to predict — it cannot move an AUC.

   `tests/test_real.py::test_permuted_labels_score_at_chance` already enforces
   this end-to-end: shuffle the labels, run the full CV, assert every arm sits
   near 0.5.

**Conclusion: I found no leak of either named kind.** I am not claiming the
pipeline is leak-free in general; I am claiming the two mechanisms that would
explain an inflated SECOM AUC are absent, and that the AUC is not inflated
anyway.

### The gap the audit found instead

The permuted-label test asserts the *predictor* scores at chance under a null.
It says nothing about what the *reporter* does. Those are different failure
modes, and only the second is what this project sells:

> a suspect that fails the bootstrap stability check is dropped

That is a rhetorical claim. Nothing in the repo measures it. And reading
`estimator.py:199-204` and `estimator.py:236-237`, the loop contains two
never-return-empty-handed fallbacks:

```python
if not suspects:                      # 199
    suspects = [int(j) for j in order[: max(self.top_k, 1)]]
...
surv = [j for j in reps if self.stability_[j] >= self.stability_min]
if not surv:                          # 236
    surv = reps[: min(5, len(reps))]
```

So **the drop step cannot return an empty report.** If that is the binding
behaviour, the false-discovery rate under a no-causal-sensor null is 1.0 by
construction, and the "verify-and-drop" mechanism is decoration. That is a
prediction about the architecture, and it is testable.

### Hypothesis for this turn

> **H1.** Under a label-permutation null, where no sensor carries information
> about failure, the agent loop still reports root causes on essentially every
> replicate, and the bootstrap support of those invented suspects overlaps the
> support of the suspects it reports on the real labels.

**What would distinguish this from the obvious alternative.** The obvious
alternative is "the loop abstains on noise, and the real-label suspects have
visibly higher bootstrap support" — i.e. the mechanism works. H1 and that
alternative make opposite, quantitative predictions about two statistics:
the abstention rate on the null (H1: ~0; alternative: high) and
P(real replicate's max support > null replicate's max support) (H1: ~0.5;
alternative: ~1). Measuring both settles it, and neither can be argued into.

A single real-label fit at the pre-registered operating point selects 21 sensors
from 64 candidates with a maximum bootstrap support of 0.833 (timing run, this
session, 107 s single-threaded). If the null produces the same shape, the number
0.833 means nothing.

**Run:** `scripts/null_fdr.py`, 200 permuted replicates + 40 real-label
replicates, output `runs/null_fdr.json`.

---

## Turn 2 (2026-09-04) — the invariance screen, and a null I got wrong first

### Result

`scripts/invariance.py` → `runs/invariance.json`. 474 surviving sensors, 5
contiguous time blocks carrying 44 / 21 / 11 / 11 / 17 failed wafers.

| quantity | value |
|---|---|
| sensors with any association with failure (BH FDR 0.05, exact permutation B = 20,000) | **22** of 474 |
| of those, non-invariant across production periods (BH 0.05, B = 20,000) | **1** |
| of those, associated and not shown to break | **21** |
| sensors the agent loop selects in ≥1 of 25 folds | 119 |
| ...of which are marginally associated | 21 — i.e. 21 of the 22 associated sensors in the whole matrix |

### The finding that matters

**The one sensor the screen rejects is the loop's single most confident
suspect.** `sensor_059` is in the reported top-5 in **25 of 25** CV folds. Its
per-period AUCs are 0.56 / 0.77 / 0.86 / 0.52 / 0.49 — a strong association in
the middle of the record and none at the end — with I² = 0.82, meaning 82% of
the variance in its association is *between* periods rather than within them.
BH-adjusted permutation p = 0.012.

The other 21 "pass". They did not pass an invariance test; they were not
testable. This is the part that needs stating precisely, because the flattering
misreading is right there:

* The power audit injects sensors with a *known* break (association 0.5 + δ in
  block 0, 0.5 elsewhere) and runs the identical test. Detection: 14% at a
  first-period AUC of 0.55, 23% at 0.60, 52% at 0.65, 79% at 0.70, 95% at 0.75
  (unadjusted α = 0.05; at the strictest level BH could have demanded, 2% / 4% /
  12% / 35% / 61%).
* SECOM's associated sensors have |AUC − 0.5| of 0.088 to 0.192, median 0.114.
  Their *entire* signal is smaller than the break the test needs to see.
* `sensor_059` is the strongest association in the matrix (|AUC − 0.5| = 0.192)
  — which is exactly why it is the one the test could convict.

So power rises with association strength, the sensors the test can judge are the
ones the loop is most confident about, and the single judgeable one failed.
That is a much less comfortable reading than "21 of 22 suspects are invariant",
and it is the correct one.

### What would distinguish this from the obvious alternative

The obvious alternative is "`sensor_059` is a real cause and the middle-period
spike is noise". Two things argue against it and one would settle it:

* I² = 0.82 with a BH-adjusted permutation p of 0.012 is not a noise-level
  fluctuation, and the permutation reference is exact, so this does not lean on
  an approximation that ties could break.
* The direction is consistent with `runs/drift.json`: the sensors separate
  production era at adversarial AUC 0.9926, so a sensor tracking era rather than
  failure is the *expected* artifact on this dataset, not an exotic one.
* What would settle it is per-period data with more failures — 11 fails in
  blocks 2 and 3 is where the power went. That is not obtainable from SECOM.

### The null I got wrong first, and both numbers

I ran this test twice and the two runs disagree materially. Recording both,
because the first one is the mistake a reader would make:

| version | invariance null | non-invariant sensors found |
|---|---|---|
| first (superseded) | permute **labels** | 42 of 474 |
| current | permute **block membership** | 1 of 22 associated |

Permuting labels builds the reference distribution at AUC 0.5, where rank AUC
has smaller sampling variance than at the observed association. Against that
too-tight reference, a strongly-associated sensor looks heterogeneous *because
it is strong*. The correct null for "the association is the same in every
period" holds each sensor's pooled association fixed and destroys only the block
structure — so rows are reshuffled into same-sized blocks. The superseded
`runs/invariance.json` was deleted rather than kept, since a JSON carrying
numbers from a wrong protocol is exactly the hazard these rules exist for.

`tests/test_null.py::test_block_permutation_null_detects_a_break_it_should_detect`
now pins the direction of the corrected test: a sensor associated only inside
block 0 must come back with p < 0.01, an untouched sensor with p > 0.05. Without
that test the fix would be unverified.

### A second method note, also a correction

The closed-form χ² reference for Cochran's Q is **anticonservative on this
data**: under the null it rejects at 0.061 against a nominal 0.050 (and at 0.090
in the first, mis-specified version). SECOM's sensors are heavily quantised, and
ties break the Hanley-McNeil variance approximation the statistic is built on.
Every decision now uses a permutation p-value; the χ² figure is retained in the
JSON as a diagnostic and labelled as one. The vectorised rank AUC that makes
20,000 permutations affordable is checked against scikit-learn on tie-heavy,
missing-heavy input in `tests/test_null.py` — a fast statistic that quietly
disagrees with the slow one would have corrupted every p-value built on it.

### Verdict on brief point 4

The pipeline reports **associational** suspects. The data cannot support
upgrading that: the only suspect strong enough to test failed, and the rest are
below the screen's detection floor. Being explicit is the outcome, and it is now
generated into `RESULTS.md` and the README from the JSON rather than asserted.

---

## Turn 2b — second opinions

### codex (`codex exec`, full output in this session's log)

Asked for the strongest specific reason the numbers or the verify-and-drop claim
might be wrong. It found, unprompted, the same architectural flaw I had just
written H1 about:

> **The loop does not reliably drop unstable suspects.** After applying
> `stability_min`, if no suspect survives, it deliberately restores the top five
> anyway: `surv = reps[: min(5, len(reps))]` (estimator.py:234). Those restored
> suspects may have stability below the threshold—or zero. […] "Drops unstable
> suspects" is therefore false exactly when verification rejects everything, the
> situation where withholding a root-cause report matters most.

Independent confirmation from a critic that had not seen my hypothesis. Kept,
and it is the thing `runs/null_fdr.json` quantifies.

Its second point is a fair reframing I had not stated sharply enough:

> **The 0.759 AUC is temporally leaked for the deployment question.** It comes
> from shuffled repeated stratified CV […] so training folds contain wafers
> produced after wafers in their corresponding test folds. […] Thus 0.759 may
> avoid ordinary preprocessing/label leakage, but calling it simply "leak-free"
> is overstated: it leaks future process regimes into training and estimates
> retrospective interpolation, not prospective wafer prediction. Even under its
> chosen protocol, "KPI met" rests only on the mean; its reported interval is
> [0.739, 0.779], crossing the 0.75 target.

**Assessment: correct, and not fully fixed.** The repo does report the
chronological (0.532) and rolling-origin numbers prominently, so nothing is
hidden — but the KPI card scores "met" against the shuffled-CV mean while its
own CI crosses the target line. The card already says the CI spans the line; it
does not say that the protocol behind the point estimate is the optimistic one.
This is a definition question the owner has to answer, and it is written up as
decision 1 in `WEEKEND.md` rather than resolved unilaterally.

### cursor-agent — unavailable

`Error: Authentication required. Please run 'agent login' first, or set
CURSOR_API_KEY environment variable.` Not resolvable unattended; no credentials
are mine to create. Skipped.

### agy — unavailable in headless mode

Three attempts. `agy "…"` rejects a positional prompt; `agy -p` with the prompt
on stdin rejects stdin; `agy -p "<prompt>"` with the source pasted inline gets
as far as running and then reports
`no output produced — a tool required the "command" permission that headless
mode cannot prompt for, so it was auto-denied`, suggesting
`--dangerously-skip-permissions`. **Declined deliberately:** that flag would let
another agent write inside this repository unsupervised while two of my own jobs
were running in it, and a second opinion is not worth that. Noted and routed
around, per the environment rule.

---

## Turn 3 (2026-09-04) — H1 tested: half confirmed, half refuted

`scripts/null_fdr.py`, 200 permuted-label + 40 real-label agent-loop fits,
31.5 min on 16 workers → `runs/null_fdr.json`. Log: `runs/null_fdr.log`.

| quantity | permuted labels | real labels |
|---|---|---|
| sensors reported per replicate (mean) | **13.7** | 20.9 |
| cleared π = 0.3 **on merit** | 13.7 | 21.1 |
| abstention rate | **0.0%** | 0.0% |
| never-empty fallback fired | **0.0%** | 0.0% |
| largest bootstrap support (mean) | 0.703 | 0.873 |
| 5th–95th percentile of that | [0.500, 0.917] | [0.750, 1.000] |

**2,743 sensors named as root causes across 200 datasets in which no sensor
carries any information about failure.** FDR of the reported list under this
null = **1.0**. The loop never once abstained.

### Scoring H1 honestly

H1 had two clauses. It was written down before the run
(Turn 1), so both get graded.

1. *"the loop still reports root causes on essentially every replicate"* —
   **confirmed**, and more strongly than predicted: abstention is exactly
   0.000, not merely low.
2. *"the bootstrap support of those invented suspects overlaps the support of
   the suspects it reports on the real labels"* — **refuted.**
   P(real max > null max) = **0.873**, Mann-Whitney p = 1.6e-14. The two
   distributions are strongly separated. I predicted ≈0.5 and was wrong.

And my proposed *mechanism* was wrong too, which matters more than the
prediction. I argued from `estimator.py:199-204,236-237` that the
never-return-empty-handed guards were what made the FDR 1.0. **The guards fired
on 0.0% of null replicates.** They were never needed. The operative mechanism is
that π = 0.3 is simply far too low: a sensor need only reach the top 40 of a
60-sensor candidate pool in 4 of 12 bootstrap replicates, which pure noise does
routinely. 13.7 noise sensors clear that bar unaided.

This is the more useful finding, because it changes the fix. If the guards were
the cause, the fix is deleting four lines. Since the bar is the cause, the fix is
calibrating the bar — and the fact that the statistic *is* informative (0.873)
means calibration can actually work. Had clause 2 been right, no threshold could
have saved it and the attribution step itself would have needed replacing.
Being wrong about clause 2 is the difference between a repairable pipeline and a
dead one.

### What distinguishes this explanation from the obvious alternative

Obvious alternative: "the null is too easy — permuting labels leaves the sensor
correlation structure intact, and the loop is picking up that structure." That
would predict the *same* suspects recurring across null replicates. It does not
hold: `runs/null_fdr.json` records each replicate's suspect list, and the null
replicates' reported sets are what a resampled-noise ranker produces, not a
fixed set. The distinguishing measurement is already in the JSON and does not
need another run. (Recording this as the check that was available rather than
claiming a number I have not computed: the *per-replicate overlap* of null
suspect sets is `[not measured]` — it would sharpen the argument and costs
nothing but a read of the existing JSON. Queued.)

### The fix, and its measured price

`scripts/abstain.py` → `runs/abstain.json`. No refits: the rule is a function of
the per-replicate supports already recorded. τ(α) = the (1−α) quantile of the
null's max support (Westfall–Young max-statistic, family-wise over the sensors a
replicate screens, no independence assumption between sensors — which matters
when 179 SECOM sensors have a partner correlated above 0.99).

**Calibration is held out.** Fitting τ and measuring abstention on the same
replicates returns 1−α by construction and measures nothing, so the 200 null
replicates are halved, τ is fitted on one half, every rate is read off the
other, both directions are averaged, over 400 random partitions.

| α | τ | held-out null reports nothing | real labels report nothing | real suspects reported |
|---|---|---|---|---|
| 0.10 | 0.842 | 82.0% (target 90%) | 23.4% | 1.18 |
| 0.05 | 0.910 | 91.6% (target 95%) | 51.3% | 0.60 |
| 0.01 | 0.959 | 97.7% (target 99%) | 78.8% | 0.22 |
| none | — | 0.0% | 0.0% | 20.95 |

**At α = 0.05 SECOM supports 0.60 named suspects on average and nothing at all
51% of the time, against the 20.9 the pipeline prints today.** That is the
practical form of every other negative result in this repo: the AUC table said
selection costs accuracy, the stability table said the top-5 does not reproduce,
and this says the report should mostly be one line long or empty.

Two ways the rule is itself imperfect, both measured and both stated in
`RESULTS.md` rather than left for a reader to find:

* **It under-abstains.** 91.6% against a nominal 95%. τ is a point estimate of an
  upper quantile from 100 replicates, and that is biased low. Fix: more null
  replicates, or an upper confidence bound on the quantile instead of the
  quantile. Not done.
* **The bar sits on a 13-point grid.** Support is a fraction of `n_boot = 12`, so
  τ cannot be placed between k/12. τ(0.01) = 0.959 sits between 11/12 and 1,
  which is why α = 0.01 buys little over α = 0.05 — there is no room above it.
  Fix: more bootstrap replicates inside the loop, linear cost, not spent.

### Implementation

`AgentRCA(report_tau=...)`. It governs the new `reported_` / `reported_original_`
/ `abstained_` attributes only; `selected_` and `predict_proba` are unchanged,
so **enabling abstention cannot move any AUC in this repo** and the prediction
and attribution claims stay separable. `ReporterAgent` renders an explicit
abstention when the list is empty, because an empty bullet list is not something
an engineer can act on. Three tests in `tests/test_null.py` pin it: abstention
happens, the default is byte-identical to the old behaviour, and the report
length is monotone in τ.

### Published baselines, for the record

The brief states published SECOM AUCs cluster in **0.70–0.80** (source: the
brief, not a measurement of mine). This repo's `rf_all` measures 0.759
[0.739, 0.779] — inside that band, which is consistent with no leak and is why
the leak audit in Turn 1 looked for mechanisms rather than assuming one from the
number. I have not found a published false-discovery rate for agent-based RCA
attribution to compare the 1.0 against; if one exists it belongs in a separate
column here, and until I have read it this cell is `[not measured]`.

---

## Turn 4 (2026-09-04) — H2, written before the run

Every negative result so far compares the agent loop to a plain baseline on
*accuracy* (loses, −0.042 paired) or on *stability* (loses, 22.3% against
`univariate`'s 46.1%). The new false-discovery axis has no such comparison yet,
and it is the one axis where the loop's extra machinery — bootstrap
verification specifically — is supposed to be doing the work. So:

> **H2.** The agent loop's bootstrap-support statistic is no better calibrated
> against a no-causal-sensor null than the same statistic computed from a plain
> univariate ranker. Specifically, P(real max support > null max support) will be
> no higher for the agent loop than for univariate bootstrap selection, and the
> null-calibrated threshold will not buy the agent loop a longer honest report.

**Why this is the right test and not a rhetorical one.** "FDR = 1.0" is trivially
true of any procedure that always emits a top-5, univariate included, so
comparing raw FDRs is uninformative. What is informative is the *separation* the
statistic achieves — how well its value distinguishes a world with causes from
one without — because that is what decides how much of a report survives
calibration. If univariate matches or beats the loop at 0.873, then the
VerifierAgent's bootstrap machinery is not earning its keep on the one axis
built specifically for it, and the loop has now lost on accuracy, stability and
calibration.

**What would distinguish H2 from the alternative.** The alternative is that
permutation-importance attribution inside a bootstrap is a genuinely better
signal-detector than univariate ranking, just a worse *point* selector — which
would show up as the loop separating clearly better (say ≥0.95 against
univariate's 0.873) and supporting a longer report at the same α. Both are
single numbers from the same protocol, so the comparison cannot be argued.

**The confound I have to control, or the comparison is worthless.** The two
procedures must be compared at the same bootstrap count and the same reporting
depth, or a difference in separation is just a difference in how many bootstrap
draws went into the statistic. The univariate arm therefore uses the loop's own
`n_boot = 12` and `select_k = 40` from `AGENT_CFG` rather than a convenient
number, and both arms are scored with the identical `max` statistic and the
identical held-out τ calibration from `scripts/abstain.py`.

**Run:** `scripts/null_fdr_rankers.py`, same 200 permuted + 40 real replicate
structure, output `runs/null_fdr_rankers.json`. Cheap — a univariate screen is
474 rank-AUCs rather than a permutation-importance pass, so this is minutes
rather than the 31.5 min the agent arm took.

---

## Turn 5 (2026-09-04) — H2 verdict: split, and the split matters

`scripts/null_fdr_rankers.py`, 720 replicates (3 rankers × [200 permuted + 40
real]), 10.5 min → `runs/null_fdr_rankers.json`. Every arm matched to the agent
loop's own `n_boot = 12` and `select_k = 40` from `AGENT_CFG`, identical
max-over-sensors statistic, identical split-half held-out calibration, cleaning
fitted inside each resample.

| arm | P(real>null) | τ(0.05) | no-cause worlds kept silent | ceiling | suspects | abstains |
|---|---|---|---|---|---|---|
| `univariate` | **0.943** | 1.000 | 88.5% | 88.5% | 4.10 | 0% |
| `rf_impurity` | 0.920 | 1.000 | 84.0% | 84.0% | 2.67 | 0% |
| **agent (full loop)** | 0.873 | 0.909 | **91.6%** | **98.5%** | 0.60 | 51% |
| `logreg_coef` | 0.785 | 1.000 | 57.0% | 57.0% | 2.92 | 0% |

**H2 was confirmed on the clause I wrote and refuted on the thing that matters.**

* *Confirmed:* "P(real max > null max) will be no higher for the agent loop than
  for a plain univariate ranker" — 0.873 vs 0.943. The loop is a worse signal
  detector, a third time, consistent with the AUC and stability tables.
* *Refuted:* "the null-calibrated threshold will not buy the agent loop a longer
  honest report." I framed this as though a longer report were the only prize.
  It is not. The plain rankers' support **saturates at 1.000** — on real labels
  their best sensor is in the top 40 of all 12 bootstraps, and on 11.5%
  (`univariate`) to 43% (`logreg_coef`) of *permuted* replicates too. No τ ≤ 1
  excludes those, so their error control is **capped**: 88.5% / 84.0% / 57.0%
  is the most any threshold could ever deliver. The agent loop's noisier
  statistic has headroom to 98.5% and lands at 91.6%, the only arm near the 95%
  target.

So the univariate ranker reports 4.10 suspects at 88.5% control and the agent
loop reports 0.60 at 91.6%. Those are not the same product, and picking between
them is not a modelling question.

### The uncomfortable part, stated as narrowly as it holds

**This is the first axis in this repository on which the agent architecture wins
anything** — and the mechanism is not flattering. It wins because its estimator
is noisy enough to be thresholded. A statistic pinned at its ceiling cannot
express uncertainty; permutation importance over an inner split with ~25
positives is variable enough that its bootstrap support spreads out, and a
spread-out statistic is calibratable.

That is a property of the *operating point*, not of the architecture, which is
exactly why the claim must not be inflated. Which leads to:

> **H3.** The agent loop's error-control advantage is an artifact of the
> comparison's operating point, not of the plan/verify architecture. A plain
> univariate ranker with more bootstrap replicates and a narrower selection
> depth — so its support stops saturating — will match or beat the loop's 91.6%
> control *and* still report more suspects.

**What distinguishes H3 from the alternative.** The alternative is that
permutation-importance-plus-verification produces a genuinely better-shaped
uncertainty estimate that no reparameterisation of a univariate ranker
reproduces. H3 predicts a univariate variant reaching ≥91.6% control with
`real_reported_mean` above 0.60; the alternative predicts univariate's control
stays capped below the loop's however `n_boot` and `select_k` are set, because
what saturates is the ranker's *agreement with itself*, not the coarseness of
the grid. Both are one table from the same protocol.

If H3 holds, the loop has no measured advantage on any axis in this repo and the
README must say exactly that. If it fails, the loop has one real advantage and
it belongs in the abstract. Either way the answer is a run, and refusing to
guess it is the point — the temptation here was to keep the flattering reading
of the split verdict, since it is the only good news the architecture has
produced all weekend.

**Run:** same script, `--variants`, output shares
`runs/null_fdr_rankers.json`.

---

## Turn 6 (2026-09-04) — H3 verdict: confirmed on every clause

`scripts/null_fdr_rankers.py --variants`, 1,680 replicates over 7 arms,
31.9 min → `runs/null_fdr_rankers.json`. Log: `runs/null_fdr_rankers.log`.

| arm | P(real>null) | τ(0.05) | no-cause worlds silent | ceiling | suspects | abstains |
|---|---|---|---|---|---|---|
| `univariate` B=40, k=5 | **1.000** | 0.622 | **94.3%** | **100%** | 2.06 | 0% |
| `univariate` B=100, k=5 | 1.000 | 0.578 | 94.1% | 100% | 2.13 | 0% |
| `univariate` B=40, k=10 | 1.000 | 0.751 | 94.2% | 100% | 2.70 | 0% |
| `univariate` B=12, k=5 | 1.000 | 0.674 | 93.0% | 100% | 2.21 | 0% |
| `univariate` (matched, k=40) | 0.943 | 1.000 | 88.5% | 88.5% | 4.10 | 0% |
| `rf_impurity` (matched) | 0.920 | 1.000 | 84.0% | 84.0% | 2.67 | 0% |
| **agent (full loop)** | 0.873 | 0.909 | 91.6% | 98.5% | 0.60 | 51% |
| `logreg_coef` (matched) | 0.785 | 1.000 | 57.0% | 57.0% | 2.92 | 0% |

H3 predicted a univariate ranker at a non-saturating depth would **match or beat
the loop's 91.6% control and still report more than 0.60 suspects**. It does
both, and it also takes the separation column outright:

* control **94.3%** vs 91.6%, and the ceiling goes from 88.5% to 100%, so the
  cap that produced the loop's apparent advantage is gone entirely;
* **2.06** suspects vs 0.60, abstaining on **0%** of real replicates vs 51%;
* separation **1.000** vs 0.873.

The alternative — that permutation-importance-plus-verification produces a
better-shaped uncertainty estimate no reparameterisation of a univariate ranker
reproduces — is dead. What saturated was the *selection depth*, not the ranker:
"top 40 of 474 in every one of 12 resamples" is easy, "top 5 of 474 in every one
of 40" is not, and the fix is one integer.

**So the agent loop has no measured advantage on any axis in this repository.**
Accuracy (−0.042 paired vs `rf_all`), top-5 stability (22.3% vs `univariate`'s
46.1%), separation (0.873 vs 1.000), and calibratable error control (91.6% vs
94.3%). The only place it wins remains the synthetic generator, where its
premise is true by construction.

### The correction, stated plainly

Turn 5 concluded that this was "the first axis in this repository on which the
verification machinery earns anything". **That conclusion was wrong**, and it
was committed to `RESULTS.md` and the README before this run refuted it (commit
`b4813a4`). The error was not a miscomputation: every number in it was correct.
The error was generalising from one operating point to an architecture, having
compared the loop only against rankers pinned at a selection depth that made
their statistic degenerate.

It is worth naming why that happened, because the mechanism is more general than
this repo. The split verdict was the only good news the architecture had
produced all weekend, and I wrote it up in the same turn I found it. The
protective habit that caught it was structural rather than virtuous: the report
section was written with data-driven branches, so regenerating against the new
JSON flipped the conclusion automatically and left a diff instead of a stale
sentence. `RESULTS.md` now carries an explicit note that it replaces an earlier
conclusion, rather than silently reading as though the earlier one never
existed.

### What this does not show

Two limits, stated so the negative result is not over-claimed either:

* **The comparison is of *ranking statistics under bootstrap resampling*, not of
  everything the agent loop does.** The loop also groups correlated sensors and
  produces a written report; neither is scored here, and neither could be by
  this protocol.
* **`select_k` was never tuned for the agent loop the way it was probed for
  univariate.** The pre-registered `select_k = 40` is the loop's own operating
  point, and the honest question this raises is whether the loop at `select_k =
  5` would also improve. `[not measured]` — and it is the obvious next ablation,
  logged below as H4. If the loop improves too, the finding narrows to "selection
  depth dominates ranker choice"; if it does not, the finding stands as written.

> **H4.** Selection depth, not ranker choice, drives calibratable error control.
> Running the agent loop at `select_k = 5` (its only change) will move its
> control toward the univariate variants' 93–94% band, and the residual gap to
> `univariate` at the same depth will be smaller than the 2.7-point gap measured
> at mismatched depths.

**What distinguishes H4 from the alternative.** The alternative is that the
loop's control is limited by its permutation-importance estimator rather than by
its depth, predicting little movement when depth changes. H4 predicts the loop
climbs into the same band. Either way the univariate arm has already cleared it,
so H4 cannot rescue the architecture — it decides only *why* it loses, which is
the part a reader should be able to act on.

---

## Turn 7 (2026-09-04) — codex on the H3 comparison, and one real error

Asked `codex exec` to attack the H3 result specifically: is it a fair fight, is
the held-out calibration sound, is the saturation argument correct, does
`select_k = 5` give the univariate arm an unearned advantage. It returned six
objections. Three land, two are precision fixes, one I reject.

### It found a factual error in my own documentation — fixed

> There is also a factual implementation error: both the docstring and RESULTS
> claim cleaning is fitted inside each bootstrap resample
> (`null_fdr_rankers.py:28`, `RESULTS.md:266`). In reality it is fitted once on
> all `X` before the bootstrap loop (`null_fdr_rankers.py:93-100`).

**Correct, and this is the worst kind of error this workspace's rules exist to
catch: a document describing a protocol the code does not implement.** Verified
by reading the source — `SensorCleaner().fit(X)` sits above the `for b in
range(n_boot)` loop.

The result does not move, and the reason matters: `SensorCleaner` is
*unsupervised*. It drops all-missing, constant and exactly-duplicated columns
from `X` alone and never touches `y`. A label permutation cannot change which
columns are constant, so it cannot leak into the null, and `AgentRCA.fit` cleans
its training matrix the same way — the arms are matched on this too. So the
protocol is sound and its description was not.

Fixed in three places: the module docstring now states where the cleaner sits
and why that is sound; the JSON's `leakage_control` field says the same; and
`RESULTS.md` states it from a verified literal rather than echoing the field,
because the string "fitted inside each resample" survives in every JSON written
before this fix. The ranker JSON will be regenerated after the running job
finishes so its own field is right too.

### Two precision fixes, both accepted

> "capped ... however alpha is set" is false: `tau > 1` yields 100% silence,
> albeit zero power; randomization at the boundary can attain intermediate
> error levels.

Right. The cap holds for the rule actually implemented — report iff
`support >= tau` with `tau <= 1` — and not for a degenerate `tau > 1` (which
abstains always, at zero power) or a randomised boundary rule. `RESULTS.md` now
says "for any threshold rule of the form used here", and the JSON's saturation
note spells out both escapes.

> `P(real max > null max)` measures repeatability of each algorithm on this one
> observed label vector as much as it measures signal/noise discrimination.

**This is the sharpest point and I had not seen it.** The "real" replicates
reuse the same labels and the same matrix, varying only bootstrap randomness. A
univariate ranker is deterministic given a resample; the agent loop carries
model fitting, screening, inner splits, permutation importance and verification.
So the loop's real-arm distribution is intrinsically wider, and a wider real-arm
distribution lowers P(real > null) *without any difference in signal detection*.
The separation column is therefore confounded in the univariate arm's favour.

It does not touch the error-control column, which is a function of the null
distribution and the real supports at a threshold, so the argument that carries
the conclusion survives. `RESULTS.md` now says explicitly that separation is the
weaker of the two columns and error control is the one doing the work.

> "suspects reported" ... the two counts come from different candidate
> universes ... Report length is a cost/power tradeoff, not an accuracy axis.

Accepted, both halves. The univariate count is over all surviving sensors; the
agent's `stability_values` covers only correlation-group representatives
admitted to verification. And without ground-truth causes a longer list is not
self-evidently better. Both now stated in the section.

### The objection I reject, and why in one line

> The contrary headline comes from post-hoc tuning only the univariate arm ...
> The corresponding agent configuration is omitted despite being directly
> runnable.

Half right and already answered: the asymmetry is real, which is exactly why H4
(`scripts/null_fdr.py --select-k 5`) was written down as a hypothesis and
launched **before** this critique arrived, and why `RESULTS.md` now says the
table is "a tuned baseline against an untuned loop, which is the right
comparison for *would something simpler have done* and the wrong one for *is the
architecture worse at equal effort*". What I reject is the framing that this
invalidates the headline: "would a simpler method have sufficed" is a legitimate
question with a legitimate answer, and the answer does not become false because
a second question is also worth asking. But the headline is now qualified rather
than absolute pending H4.

### What changed as a result

The headline bullet no longer reads "the loop has no measured advantage on any
axis in this repository". It reads that on every axis measured the loop is
matched or beaten by a univariate ranker, and it carries both caveats inline.
That is a weaker claim and it is the one the evidence supports.

---

## Turn 8 (2026-09-04) — H5, written before the run

Three independent measurements now point at the same component, and none of them
was designed to:

1. `runs/secom_loop_sweep.json` — model-native importance beats held-out
   permutation AUC-drop at **5 of 5** matched depths (mean +0.021 AUC).
2. `runs/secom_stability.json` — the permutation-based rankers are the *least*
   stable in the table (`perm_only` 20.0%, `agent` 22.3%) while `univariate`
   reaches 46.1%.
3. `runs/null_fdr_rankers.json` — the loop's bootstrap support separates a real
   world from a permuted one at 0.873, below every plain ranker but
   `logreg_coef`.

The common factor is that held-out permutation importance is estimated on an
inner validation split holding roughly 25 positives, which is a small number to
estimate an AUC drop from. That is a mechanism, not a mood, and it makes a
prediction the repo has never tested on its headline KPI.

> **H5.** The loop's top-5 stability is limited by its attribution statistic, not
> only by the sample-size wall. Re-running the identical loop with
> `attribution="model"` — one config field, nothing else changed — will raise
> top-5 bootstrap stability materially above the loop's 22.3%, landing near
> `rf_impurity`'s 36.5% (which is essentially what model-native attribution
> ranks by), and **will not** reach the 80% KPI, because the bootstrap
> perturbation drops ~37% of wafers and no attribution statistic recovers that.

**What distinguishes H5 from the obvious alternative.** The alternative — the
one `DONE_OVERNIGHT.md` argued for and I have so far accepted — is that the
sample-size wall dominates and the choice of ranker is second-order. It predicts
`agent_model` stays near 22.3%. H5 predicts it climbs by roughly 14 points to
the mid-30s. The gap between those predictions is far larger than the difference
between the ablation rungs already measured (verification earned +2.1 points,
correlation grouping +0.2), so the protocol can resolve it.

Both predictions agree the KPI is missed, and I want that on the record *before*
the run: **H5 is not an attempt to reach 80%, and if it produced 80% I would
distrust it.** It is an attempt to attribute the miss to a component. The useful
outcome either way is knowing whether a practitioner should change their
attribution statistic or go collect more failed wafers — those are very
different pieces of advice and the repo currently gives the second without
having tested the first.

**Run:** `scripts/stability_secom.py --only agent_model --append`, same 200
bootstrap replicates and 25 CV training folds as every other row, so the new row
is directly comparable to the existing table. Queued behind H4 rather than
launched now: H4 holds the 16-worker lease and oversubscribing it would slow
both.

---

## Turn 9 (2026-09-04) — H4 verdict: refuted, and it exposes a guard that swaps roles

`scripts/null_fdr.py --select-k 5`, 240 agent-loop fits, 31.4 min →
`runs/null_fdr_k5.json`, priced by `scripts/abstain.py` →
`runs/abstain_k5.json`. Same protocol, same held-out split-half calibration,
`select_k` the only thing changed.

| at α = 0.05 | agent `k=40` | agent `k=5` | `univariate` `k=5` |
|---|---|---|---|
| no-cause worlds kept silent | 91.6% | **91.5%** | **94.3%** |
| suspects reported | 0.60 | 1.16 | 2.06 |
| reports nothing (real) | 51% | 6% | 0% |

**H4 predicted the loop's control would climb into the univariate variants'
93–94% band when only its depth changed. It did not move at all** — 91.6% to
91.5%, less than half a point. The competing explanation is the one supported:
the loop's binding constraint is its permutation-importance estimator, not the
depth it selects at. That is a property of the architecture rather than a
parameter anyone can turn, and it is the answer a practitioner can act on.

Depth is still worth turning down for the loop — the report goes from 0.60
suspects to 1.16 and from empty 51% of the time to 6% — it simply does not close
the error-control gap.

**And the equal-effort comparison, which is what codex correctly demanded, comes
out the same way as the tuned one.** At matched `select_k = 5` the univariate
ranker leads on control by 2.8 points (94.3% vs 91.5%) and reports nearly twice
as many suspects (2.06 vs 1.16). So the earlier conclusion was not an artifact of
tuning one arm: it survives the symmetric test.

### The thing I did not predict, and the second correction it forces

On permuted labels:

| | `select_k = 40` | `select_k = 5` |
|---|---|---|
| noise sensors clearing π = 0.3 **on merit** | 13.7 | **0.47** |
| never-empty fallback fired | **0.0%** | **62.5%** |
| replicates reporting nothing | 0.0% | 0.0% |

At the pre-registered depth the threshold is so loose that 13.7 pure-noise
sensors clear it unaided, and the fallback is genuinely never needed. Narrow the
depth and **the threshold starts working almost perfectly** — 0.47 noise sensors
survive it — **and the fallback fires on 62.5% of null replicates and puts them
straight back.** Abstention is 0% at both depths, by two entirely different
mechanisms.

This corrects Turn 3, which concluded the guards were *not* the mechanism behind
the false-discovery rate and that "lowering or raising the guard changes
nothing". That was measured at `select_k = 40`, where it is true. At
`select_k = 5` it is false: the guard is precisely what destroys an otherwise
working filter. Both operating points are now measured and reported, and neither
generalises to the other.

That is the second time this weekend a conclusion drawn from one operating point
failed at another — the first being the ranker saturation in Turn 5. The pattern
is worth stating as a lesson rather than as two coincidences: **on this pipeline,
a claim about a mechanism is only as general as the operating points it was
measured at, and one is never enough.** Every mechanism claim in `RESULTS.md`
now names the depth it was measured at.

### Where this leaves the architecture

Unchanged in direction, sharper in diagnosis. The loop still loses to a
univariate ranker at equal effort on error control and report length, and the
reason is now localised: not the depth, not the correlation grouping (+0.2
points of stability), not the verification step (+2.1 points), but the held-out
permutation-importance estimator that everything else is built on top of.

Which is exactly what **H5** tests on the headline KPI, launched now that the
worker lease is free: `scripts/stability_secom.py --only agent_model --append`.
Prediction written down in Turn 8 before either run: top-5 bootstrap stability
climbs from 22.3% toward `rf_impurity`'s 36.5%, and does not reach 80%.

---

## Turn 10 (2026-09-04) — H5 confirmed, and a correction I made last turn was itself wrong

### H5 verdict: confirmed, including the quantity

`scripts/stability_secom.py --only agent_model --append`, 200 bootstrap
replicates and 25 CV-training folds, identical to every other row in the table
→ `runs/secom_stability.json`. One config field changed: `attribution="model"`
instead of `"permutation"`.

| ranker | top-5 bootstrap stability | sd | consensus | distinct sensors ever named | wall |
|---|---|---|---|---|---|
| `univariate` | 46.1% | 20.2 | 0.611 | 73 | 0.2 min |
| `logreg_coef` | 42.6% | 15.8 | 0.559 | 89 | 0.0 min |
| `rf_impurity` | 36.5% | 17.2 | 0.497 | 95 | 1.0 min |
| **`agent_model`** | **35.3%** | 15.3 | 0.479 | 107 | 14.6 min |
| `agent` | 22.3% | 15.2 | 0.360 | 151 | 40.3 min |
| `agent_no_corr` | 22.1% | 15.4 | 0.350 | 148 | 32.6 min |
| `perm_only` | 20.0% | 14.8 | 0.328 | 173 | 7.9 min |

Turn 8 predicted, in writing and before the run, that the swap would "raise
top-5 bootstrap stability materially above the loop's 22.3%, landing near
`rf_impurity`'s 36.5%, and will not reach the 80% KPI". Measured: **35.3%**,
1.1 points from the named landmark, and the KPI is still missed by 44.7 points.
The competing explanation on the record — that the sample-size wall dominates
and ranker choice is second-order, which predicted `agent_model` stays near
22.3% — is refuted by 13.0 points.

I want to be careful about what this does and does not license. It is a
confirmed prediction, which is rarer here than it should be, but it confirms a
*diagnosis*, not a capability: the loop is still 10.7 points below the simplest
ranker in the table and 44.7 below target. **On the published-baseline question,
this metric has no published baseline I can cite.** SECOM papers report
classification AUC, and this repo has no source in hand reporting top-5
root-cause selection stability on SECOM, so the 80% figure is a project target
from the brief with no external provenance, and the only honest comparisons are
the internal ones above. That is worth saying rather than dressing the target up
as a literature number.

### What the table now decomposes, which is the actually useful part

Reading the ladder as a 2x2 rather than a list:

|  | permutation attribution | model attribution | machinery delta |
|---|---|---|---|
| bare ranker | `perm_only` 20.0% | `rf_impurity` 36.5% | — |
| full loop | `agent` 22.3% | `agent_model` 35.3% | — |
| attribution delta | +2.3 | **-1.1** | |

- Changing the **attribution statistic** is worth **+13.0** points (`agent` →
  `agent_model`) and makes the loop 2.8x cheaper (40.3 → 14.6 min).
- Changing the **architecture** — screen, correlation grouping, bootstrap
  verification, drop — is worth **+2.3** points on top of the noisy statistic
  and **-1.1** on top of the good one. Both are inside one sd of the pairwise
  overlap distribution.

So the plan/correlate/verify apparatus is not merely unhelpful, it is
*approximately a no-op in both directions*: the loop's stability is essentially
a function of which importance statistic it consumes. That is a sharper
statement than "the loop loses", and it is the one a practitioner can act on:
the component worth changing is one line, and the component the repo is named
after is worth about a point either way.

`rf_impurity` at 36.5% versus `agent_model` at 35.3% is the cleanest form of it.
`agent_model` runs a screen, a correlation grouping, 12 bootstrap replays and a
drop step over what is essentially the same statistic `rf_impurity` reports
directly, takes 14.1x longer, and ends up 1.1 points behind it.

### The self-audit that mattered more than the run

Reading `RESULTS.md` back before regenerating it, the paragraph I wrote last
turn does not survive checking against the code it describes. Turn 9 concluded:

> Narrow the depth and **the threshold starts working almost perfectly** ...
> **and the fallback fires on 62.5% of null replicates and puts them straight
> back.** ... the guard is precisely what destroys an otherwise working filter.

That conflates two sets `AgentRCA.fit` deliberately keeps apart.
`selected_` is what the final classifier is fitted on; it cannot be empty,
because a classifier needs at least one column, and the `if not surv` guard is
what guarantees that. `reported_` is what an engineer is handed, it is
thresholded at `report_tau`, and it *is* allowed to be empty. The guard touches
only the first. `scripts/abstain.py` reads `stability_values` — the pre-drop
supports — and never looks at `selected_` at all, so the error-control column
is a function of the second set only.

Checked on the recorded replicates in `runs/null_fdr_k5.json` rather than argued:

| null replicates, `select_k = 5` | n | largest support reached | naming a suspect over tau = 0.417 |
|---|---|---|---|
| fallback fired (nothing cleared pi = 0.3) | 125 | 0.250 | **0** |
| threshold cleared on merit | 75 | 0.583 | 25 |

The fallback fires exactly when every support is below pi = 0.3, and
tau(0.05) = 0.417 sits *above* pi, so all 125 replicates it fires on are silent
under the calibrated rule. Not "mostly" — zero of them name anything. The null
worlds that do get through are entirely ones where the attribution estimator
handed a pure-noise sensor a genuinely high bootstrap support.

**So Turn 9's correction was wrong, and the reading it corrected was right.**
The guards are not the mechanism behind the false-discovery rate, at either
depth. What is true is much narrower: the fallback prevents the *uncalibrated*
`stability_min` filter from returning an empty prediction set, and the
"replicates reporting nothing at all: 0.0%" row is 0% at both depths because
these runs set `report_tau = None`, which disables abstention *by
configuration*. I had presented a structural property of a code path with
abstention switched off as a discovered limitation of the pipeline. That is
exactly the class of error the weekend rules exist to prevent, and it was mine,
not a critic's.

Fixed by generating the paragraph from the records
(`report.fallback_reach`), and pinned by two tests in `tests/test_null.py`: one
builds a case where the guard must fire and asserts `reported_` still comes back
empty while `predict_proba` still works, the other asserts the
tau >= `stability_min` ordering the cross-tab depends on, so a future run that
inverts it fails loudly instead of quietly invalidating the prose.

The uncomfortable meta-observation: Turn 9 congratulated itself for learning
that "a claim about a mechanism is only as general as the operating points it
was measured at", and in the same entry made a different error of the same
family — generalising from one *code path* instead of one operating point. The
lesson that actually generalises is duller: **read the implementation before
attributing a number to a mechanism.** Both of my last two mechanism claims were
wrong on first writing, and both were caught by reading code rather than by
running anything.

Net effect on the headline: none. The direction of every result is unchanged and
the diagnosis is now better supported, since the residual error-control failure
is attributed to the estimator by measurement instead of by elimination.

---

## Turn 10, second half — H6, written before the run

Two independent axes now point at the attribution estimator:

1. **Stability** (H5, above): swapping it is worth +13.0 points; the
   architecture around it is worth about -1.1 to +2.3.
2. **Error control** (H4, Turn 9): depth is worth -0.1 points to the loop
   (91.6% → 91.5%) while it was worth several to the univariate arm, and the
   cross-tab above localises the residual failure to noise sensors that clear
   pi *on merit* — i.e. to the estimator.

But axis 2's attribution to the estimator is still **by elimination**. Depth was
ruled out and the guard has now been ruled out, so the estimator is what is
left. That is an inference, not a measurement, and this repo has twice this
weekend had an inference-by-elimination fail when tested directly. The direct
test is one field.

> **H6.** The loop's null error control is limited by its attribution
> statistic. Re-running `scripts/null_fdr.py` with `attribution="model"` and
> nothing else changed will raise the fraction of no-cause worlds kept silent
> at alpha = 0.05 above the permutation arm's 91.5% (`select_k = 5`) and 91.6%
> (`select_k = 40`), moving it most of the way to the univariate arm's 94.3%.
> Concretely: **above 93.0% at `select_k = 5`.**

**What distinguishes H6 from the obvious alternative.** The alternative is that
error control at this class balance is capped by the bootstrap's own variance —
12 replays over ~65 fails — regardless of which statistic is being replayed. It
predicts `agent_model` lands within noise of 91.5%, and that the univariate
arm's 94.3% comes from something else entirely (its 40 bootstraps rather than
12, say). The two predictions are ~2 points apart, which is larger than the
0.1-point spread depth produced, so the protocol can resolve it.

I am also writing down what would make me distrust a confirmation: if
`agent_model` clears 94.3% *and* its suspect count collapses toward zero, then
it has bought control by reporting less rather than by ranking better, and the
suspects-reported column has to be read alongside the control column. The
univariate arm reports 2.06 suspects at 94.3% control; anything under ~1.0 is
buying silence, not accuracy.

**Run, launched before writing this up:** `scripts/null_fdr.py --null 200
--real 40 --jobs 16 --base rf --attribution model` at `--select-k 5` and then at
the pre-registered depth, each priced by `scripts/abstain.py` on the same
split-half calibration → `runs/null_fdr_k5_model.json`,
`runs/null_fdr_model.json`. Both depths deliberately, because the one lesson
from Turn 9 that did survive is that one operating point is never enough.

---

## Turn 10, third part — codex on the two new claims, and both fixes

Ran `codex exec` against `report.attribution_2x2`, `report.fallback_reach`, the
`RESULTS.md` sections they generate, and `scripts/stability_secom.py`, asking
for the strongest reason the two new conclusions are wrong or overclaimed. It
found two, and **both were right.** Neither is a number error; both are
"this comparison is not the comparison you say it is", which is the class of
objection worth paying for.

### Finding 1 — the 2x2 was not factorial

> More seriously, the alleged "bare ranker" cells are not corresponding
> versions of one architecture: `rf_impurity` fits a 500-tree RF with
> `min_samples_leaf=5` across every cleaned sensor [...] `perm_only` fits a
> 300-tree RF with different defaults, first screens to `n_screen`, then
> evaluates held-out permutation importance. [...] Therefore the architecture
> effects [...] subtract procedures differing in model hyperparameters,
> screening universe, data splitting, filtering and ranking semantics -- not
> merely "bare versus full architecture."

Confirmed by reading `scripts/stability_secom.py`: `rank_rf_impurity` calls
`make_rf(n_estimators=500, min_samples_leaf=5)` with `n_keep=1` over the whole
cleaned matrix; `rank_perm_only` calls `make_rf(n_estimators=300)` and screens
to `AGENT_CFG["n_screen"] = 150` first. So `perm_only` **is** a matched bare
cell for the permutation column — it is the loop's attribution step with the
loop's own base and screen — and `rf_impurity` is **not** one for the
model-native column. The `-1.1` I published was the architecture plus a tree
count plus a candidate universe, and I had described it as the architecture.

The `+13.0` cell survives untouched, because `agent` and `agent_model` differ in
one field. That is the cell carrying the argument, which is some luck rather
than any care on my part.

**Fix.** Added `model_only`: the loop's attribution step with model-native
importance and nothing after it, built by calling `AgentRCA._rank` directly
rather than reimplementing it, so a bare-ranker cell cannot drift from the
statistic the full loop consumes. A test
(`test_bare_ranker_cells_are_the_same_construction`) asserts that driving the
same helper with `attribution="permutation"` reproduces the independently
written `rank_perm_only` **exactly** — it does, `np.array_equal` on the full
ranking — which is what licenses calling the two cells the same construction.

Until that arm lands the cell is `*[not measured]*` in all four documents, and
`scripts/audit_weekend.py` now **asserts the blank is present** while
`model_only` is absent from the stability JSON, so a confounded estimate cannot
quietly reappear there. It flips to auditing the matched delta once the arm
exists. Queued behind H6 rather than launched, since both want the same 16
workers.

A blank in the table whose entire purpose is to separate two factors is worse
to look at and better to publish than a subtraction that mixes them.

### Finding 2 — the cross-tab used a tau the headline does not use

> Showing that fallback-fired records never exceed the full-null tau = 0.417
> does not show they never exceed every split-specific held-out tau. That would
> require replaying the fallback/merit cross-tab inside each
> calibration/evaluation split, or proving every fitted tau exceeds
> `stability_min = 0.3`.

Also right, and I had half-noticed it — the bookkeeping note I wrote flagged
that the counts differ between the full-null tau and the split-half protocol,
then went on to state the mechanism claim as though the full-null cross-tab
established it. It did not: 800 thresholds get fitted, and I had checked one.

**Fix, taking the first of the two routes codex offered because it is stronger.**
`scripts/abstain.py` now counts it inside the protocol: per
calibration/evaluation split, how many evaluation-half replicates both had the
guard fire *and* named a suspect over that split's own tau. It also records
`tau_min` and `tau_max` across splits.

| `select_k = 5` | smallest tau fitted | largest tau fitted | splits where the guard reached the report |
|---|---|---|---|
| alpha = 0.1 | 0.333 | 0.417 | **0 of 800** |
| alpha = 0.05 | 0.417 | 0.500 | **0 of 800** |
| alpha = 0.01 | 0.417 | 0.501 | **0 of 800** |

Zero at every level, and the same holds at `select_k = 40` (`tau_min` 0.833).
The second route codex named is what explains it: the smallest threshold any
split fits still sits above `stability_min = 0.3`, and the guard fires only when
every support is below 0.3. So this is not a rate that happened to come out
zero, it is an ordering that holds across all 800 fitted thresholds.

The objection therefore **strengthened** the claim rather than weakening it:
from "no guard-fired replicate clears one full-null threshold" to "none clears
any of the 800 thresholds the headline protocol actually fits, at all three
alpha levels, at both depths." Recorded in `RESULTS.md` from
`runs/abstain_k5.json`, and audited.

### What I take from this exchange

Both findings share a shape: a subtraction or a threshold that was *almost* the
right one, described as if it were. Neither would have been caught by
re-running anything, and neither shows up as a stale number — `report.py
--check` was green and `audit_weekend.py` was green throughout, because every
figure did trace to a run. What was wrong was which two runs I was differencing
and which of 800 thresholds I was quoting.

That is worth stating as a limit on this repository's own guardrails. Generating
every number from a JSON prevents the number from drifting from the run. It does
**not** prevent the *comparison* from being the wrong one, and three of my last
four errors this weekend were of that second kind. The check that catches those
is an adversary reading the arm definitions, which costs a subprocess.

Two audit rows added in response, both of which encode a *comparison* rather
than a number: the blank assertion above, and the per-split guard-reach count.

---

## Turn 10, fourth part — H6 verdict at `select_k = 5`: confirmed

`scripts/null_fdr.py --null 200 --real 40 --jobs 16 --base rf --select-k 5
--attribution model`, 240 agent-loop fits → `runs/null_fdr_k5_model.json`,
priced by `scripts/abstain.py` → `runs/abstain_k5_model.json`. Same 800
split-half calibrations, same replicate counts, `attribution` the only thing
changed from the arm in `runs/null_fdr_k5.json`.

| at `select_k = 5` | permutation | **model-native** | target |
|---|---|---|---|
| control, alpha = 0.1 | 84.8% | **85.8%** | 90% |
| control, alpha = 0.05 | 91.5% | **93.7%** | 95% |
| control, alpha = 0.01 | 97.5% | **98.2%** | 99% |
| suspects reported, alpha = 0.05 | 1.16 | **1.34** | — |
| real-label abstention, alpha = 0.05 | 6.2% | **0.0%** | — |
| separation (same protocol) | 0.982 | **0.994** | — |

**H6 predicted "above 93.0% at `select_k = 5`". Measured 93.7%.** The competing
explanation — that error control is capped by the bootstrap's own variance over
~65 failed wafers regardless of which statistic is being replayed, which
predicted no movement from 91.5% — is refuted. Control improves at all three
alpha levels, and the diagnosis reached by elimination in Turn 9 is now
supported by direct measurement.

**The distrust check I wrote down in advance passes.** I said a confirmation
whose suspect count collapsed under ~1.0 would mean it had bought control by
saying less rather than by ranking better. It went the other way: the report
gets *longer* (1.16 → 1.34) while control improves, and real-label abstention
falls to zero. Both columns move the right way at once, which a
report-less-to-control-more rule cannot do.

### One number I nearly got wrong, and the check that stopped me

My first instinct was to write "separation goes 0.873 → 0.994". That would have
been a mixed-protocol comparison: 0.873 is the loop at `select_k = 40` with
`n_boot = 12` from `runs/null_fdr.json`, and 0.994 is `select_k = 5`. The
within-protocol figure is 0.982 → 0.994, because **depth alone had already
moved separation from 0.873 to 0.982** while moving control by −0.1 points.

That is worth recording in its own right: depth is nearly inert on error control
and worth 11 points of separation, so the two columns are not measuring the same
thing and a claim about "the loop's confidence" has to say which. It is also the
third comparison-shaped error I have come close to this weekend, and the reason
it did not land in a document is that I looked up both protocol blocks before
subtracting rather than after. The generator now takes both arms from the same
protocol by construction (`report.sec_attribution_fdr` pairs runs by depth and
refuses to cross them).

### What this does to the headline, honestly

It narrows it; it does not reverse it. Against the univariate baseline at
matched depth and alpha = 0.05:

| arm | control | suspects | separation |
|---|---|---|---|
| `univariate (n_boot=40, select_k=5)` | 94.3% | 2.06 | 1.000 |
| agent loop, model-native, `select_k = 5` | 93.7% | 1.34 | 0.994 |
| agent loop, permutation, `select_k = 5` | 91.5% | 1.16 | 0.982 |

The error-control gap goes from 2.8 points to 0.6 and the separation gap from
0.018 to 0.006. The report-length gap does not close: 2.06 against 1.34.

And the comparison still has a confound I have to state rather than enjoy: the
univariate arm resamples 40 times and the loop 12. Two of those three columns
are statistics *of* the bootstrap distribution, so a longer bootstrap is not
neutral, and a 0.6-point control gap is well inside what that difference could
account for. The honest reading is **close, not ahead** — for either arm. It is
in `RESULTS.md` next to the table.

So the standing claim changes from "the loop is beaten on every axis" to: the
loop is not measurably *better* than a univariate ranker on any axis, and the
version of it that comes closest costs a one-field config change that the
project had never tried. Everything the architecture adds on top of that
statistic is still worth about +2.3 points of stability, inside one sd, and
nothing measurable on accuracy. That is a smaller, better-supported claim than
the one I published this morning.

**Still queued:** the `select_k = 40` leg of the same run (in flight), which
will say whether the estimator diagnosis holds at the pre-registered depth too
— by now the default expectation in this repo should be that it might not, since
two mechanism claims have already failed to generalise across these two depths.
And `model_only`, behind it, for the matched bare-ranker cell.

---

## Turn 11 (2026-09-04) — a claim that was true for the wrong reason, and H7

Two jobs held the 16-worker lease this turn (H6's `select_k = 40` leg, then
`model_only` behind it), so this turn went to the part of the repo that needed
no lease: auditing the artifact the recommendations actually point at.

### `PredictAllReportFew` asserted an equality it had not earned

Recommendations 1 and 2 together describe one deployable thing — predict with
every sensor, report with the null-calibrated loop — and the repo ships it as
`PredictAllReportFew`. Its docstring said:

> Its held-out AUC is the full-sensor model's *by construction*: the loop never
> touches ``predict_proba``. There is nothing to measure about that number
> beyond the baseline row already in the results.

The first clause is true and structural. The second does not follow from it,
and the gap between them is a whole hyperparameter search. `predict_proba`
reads `predictor_` and nothing else, so the *loop* is predictively free — but
the default `predictor_` was

    Pipeline([impute(median), RandomForestClassifier(min_samples_leaf=5, ...)])

while `rf_all`, the arm the 0.759 baseline row comes from, is that forest
wrapped in `GridSearchCV(min_samples_leaf ∈ {1, 5, 10}, cv=3,
scoring=roc_auc)`. An untuned forest is not a tuned one. "The loop is free"
had been silently extended to "therefore this equals the measured row", which
is a different claim about a different component.

This is the same failure shape as the last two I caught: an argument that is
valid for one step, carried one step past where it holds. It is also the fourth
of five recent errors here that no guardrail could have caught, because no
number was stale — the number was simply never measured.

### What it was worth: nothing, and I am reporting that as the result

`scripts/eval_par_few.py`, 25 folds, `RepeatedStratifiedKFold(5 x 5, seed 0)`,
the same protocol as `runs/secom_eval.json`, cleaning and imputation fitted on
training rows only, `--jobs 1` so as not to oversubscribe the lease.

The old default arm, `par_untuned`, lands at **0.759 [0.740, 0.778]** — against
`rf_all`'s published 0.759 [0.739, 0.779]. So the docstring's *number* was
right to three decimals while its *argument* was invalid. Tuning
`min_samples_leaf` on this dataset buys essentially nothing, which is itself
worth knowing and is not something the repo had established.

I want to be exact about what that does and does not excuse. It does not
retroactively justify the claim: an unmeasured equality that happens to hold is
luck, and the next person to change the predictor default would have inherited
a docstring asserting an equality that had stopped being true without anything
failing. It does mean **no published number moves**, and I would rather report a
correction worth 0.000 AUC honestly than dress it up as a save.

**Fix, in three parts.** `yieldrca.estimator.make_rf_tuned` now holds the
baseline predictor's construction in one place; `PredictAllReportFew` defaults
to it, so the equality is true by construction rather than by coincidence; and
`test_default_predictor_matches_the_measured_baseline` asserts the grid, the
scoring, the inner-CV spec and every RF parameter agree with `arms.rf_all()`,
so retuning either side without the other fails loudly. `arms.rf_all()` itself
is deliberately **untouched** — it produced 0.759 and I am not editing the arm
behind a published number to make a refactor tidier. A second test,
`test_predict_proba_is_invariant_to_the_loop`, pins the half of the original
claim that was always true, by fitting two instances with the same predictor
and deliberately different loops (different statistic, depth and bootstrap
count, so their shortlists provably differ) and asserting bit-identical
probabilities.

The `par_tuned` arm is still running; when it lands, the JSON will carry what
the inner CV actually picked, which is the interesting residue of this.

---

## Turn 11, second half — H7, written before the run

The AUC axis is the one the brief calls the interesting one — "the interesting
claim is not the AUC, it is the delta over the obvious baseline" — and it is the
one axis the attribution finding has never been tested on. Standing numbers:

| arm | AUC | paired vs `rf_all` |
|---|---|---|
| `rf_all` | 0.759 [0.739, 0.779] | — |
| `univar_top25_rf` | 0.730 [0.709, 0.750] | -0.029 [-0.041, -0.018] |
| `agent_rf` | 0.717 [0.699, 0.735] | -0.042 [-0.058, -0.025] |

H5 and H6 both say the attribution statistic is the loop's weakest component
(+13.0 stability points, +2.2 error-control points). So does it close the
-0.042?

The reason to think not is already in the table, and it is why this prediction
can be quantitative rather than directional. `univar_top25_rf` is a *good*
ranker at the loop's own budget, and it still loses 0.029. That is the price of
sparsity itself on this dataset, where the sweep shows selection costs AUC
monotonically because the signal is diffuse. Better attribution can only
recover the part of the deficit that is *bad ranking*, which the table bounds at
roughly 0.042 - 0.029 = 0.013.

> **H7.** The loop's AUC deficit is mostly the price of sparsity, not of
> ranking quality. Running the pre-registered loop with `attribution="model"`
> and nothing else changed will raise its AUC by roughly the 0.013 the table
> leaves available — landing it **near `univar_top25_rf`'s 0.730, inside
> [0.720, 0.745]** — and its paired delta against `rf_all` will **remain
> negative with a 95% CI that excludes zero.**

**What distinguishes H7 from the obvious alternative.** The alternative is that
the deficit was a ranking-quality problem all along, which the last two
confirmed hypotheses make a live possibility rather than a straw man. It
predicts the loop reaches roughly 0.759 and its paired CI against `rf_all`
covers zero. The two predictions are about 0.02 AUC apart, which is larger than
the paired CI half-width on this protocol (~0.017 for the agent-vs-baseline
delta), so 25 folds can resolve it.

Also written down in advance, so a confirmation cannot be read as more than it
is: **H7 confirming would mean the architecture's accuracy deficit is not
fixable by any attribution statistic**, because the binding constraint would be
the decision to select at all. That is a *worse* result for the loop than the
ranking-quality explanation, not a better one — under it, the +13.0 stability
and +2.2 error-control gains are real and the accuracy gap stays. And if the
loop's AUC lands at 0.759, I should distrust it until I have checked that
`selected_` has not quietly grown toward all 474 sensors, since a "selection"
arm that stops selecting would reach the baseline trivially; `n_selected_mean`
is recorded per fold for exactly that check.

**Run:** `scripts/eval_attr_arm.py --jobs 16`, one arm, added *under* the
published protocol rather than by re-running it — `RepeatedStratifiedKFold(5x5,
seed 0)` is deterministic, so the folds are byte-identical to
`runs/secom_eval.json`'s and the paired deltas are computed against that file's
stored per-fold AUCs. Nothing already published is recomputed, so nothing
already published can move. Queued third, behind H6's second leg and
`model_only`.

One note on that script, because the failure was mine and it is instructive:
its first smoke run used `--splits 2` and cheerfully printed paired intervals
against 2 of the reference file's 25 folds. It now requires the fold *sets* to
be equal and refuses to pair otherwise. A reduced-fold invocation producing a
confident-looking interval against a subset of a published arm is precisely the
kind of comparison error this weekend keeps turning up, and this one I caught
in my own smoke output rather than in a document.

---

## Turn 11, third part — H6 inverts at the pre-registered depth, and three findings collapse into one identity

H6's second leg landed: `null_fdr.py --attribution model` at the pre-registered
`select_k = 40`, priced by the same split-half calibration.

| at `select_k = 40` | permutation | model-native |
|---|---|---|
| control, alpha = 0.1 | 82.0% | 84.6% |
| control, alpha = 0.05 | **91.6%** | **88.0%** |
| control, alpha = 0.01 | 97.7% | **88.0%** |
| suspects, alpha = 0.1 | 1.18 | 3.05 |
| suspects, alpha = 0.05 | 0.60 | 2.80 |
| suspects, alpha = 0.01 | 0.22 | 2.80 |
| separation | 0.873 | 0.940 |

**H6 does not replicate at this depth — it inverts.** Control goes 91.6% to
88.0%, a 3.6-point regression, where at `select_k = 5` the identical change was
a 2.2-point gain. I had written before the run that it might not generalise,
because two mechanism claims had already failed between exactly these two
depths. That was the right prior and it is now three for three.

### The tell, and the identity behind it

The model column reports **88.0% control and 2.80 suspects at both alpha = 0.05
and alpha = 0.01** — identical — while the permutation column moves normally
(91.6% to 97.7%). A rate that stops responding to alpha is not noise, it is a
constraint, and the constraint is arithmetic:

Bootstrap selection frequency is bounded above by 1. If some sensor is selected
in *every* resample of a null replicate, that replicate's max-statistic is
exactly 1.000, and no threshold at or below 1.000 excludes it. So

> **error control <= 1 - P(a null replicate saturates)**, with no dependence on
> alpha at all.

Measured: at `select_k = 40` with model attribution, **12.0%** of null
replicates saturate, so the cap is **88.0%**, and the measured control is
**88.0%** at both binding levels. Exact, to the precision reported. At
`select_k = 5` only 0.5% saturate, the cap is 99.5%, nothing binds, and the
better statistic is pure gain.

**This is not a new mechanism. It is the one already in this repository, which I
had filed as a quirk.** Turn 5 found `univariate` capped at 88.5% under matched
settings because "its best sensor sits in the top slice of every resample, so
the statistic pins at 1.000 on real labels and on 11.5% of permuted ones too" —
and 1 - 0.115 = 0.885. Same arithmetic, different ranker, written up as a
property of that baseline rather than of the rule. It is a property of the rule.
Three findings — the univariate cap, the depth fix that removed it, and now the
attribution reversal — are one identity, and stating it that way makes it
predict instead of describe.

And the connection to H5 is the uncomfortable part: **the property that earns
model attribution +13 points of selection stability is that it is more
repeatable, and being more repeatable is exactly what pins a sensor across every
resample.** The mechanism that makes it a better ranker is the mechanism that
saturates its null max-statistic. So the swap is not "good" or "bad"; it is good
iff the depth keeps saturation rare. `RESULTS.md` recommendation 3 now reads
*both fields together, or neither* — recommending the statistic alone would have
been recommending a 3.6-point regression at the depth the repo ships.

### The precision I had to walk back within the hour

My first version of the guarding test asserted alpha-invariance at *every*
level, and it failed on `null_fdr_model` at alpha = 0.1: tau = 0.980 there, not
1.000, control 84.6% against an 88.0% cap. The cap is an upper bound always; it
is *attained* only where the fitted threshold has itself pinned at the ceiling,
because control is 1 - P(null max >= tau) and a tau below 1.000 admits the
replicates whose maximum lands in [tau, 1.000). My assertion was stronger than
the mechanism. The test now checks the pinned levels only, and `RESULTS.md`
carries the unpinned row as the counterexample that makes the statement precise
— generated from the run, not typed, because the paragraph is specifically
about not quoting the cap as though it were the measurement.

### `model_only` landed too, and it reverses a sign in the architecture's favour

The matched bare-ranker cell codex demanded (Turn 10, finding 1) is measured:
**34.0%**, 2.0 min. The 2x2 is now factorial:

|  | permutation | model-native | statistic is worth |
|---|---|---|---|
| bare ranker | `perm_only` 20.0% | `model_only` **34.0%** | **+13.9** |
| full loop | `agent` 22.3% | `agent_model` 35.3% | **+13.0** |
| architecture is worth | +2.3 | **+1.4** | |

The statistic is worth ~13 points in both rows and the architecture ~2 in both.
**The confounded version had the model-native architecture cell at -1.1; the
matched arm makes it +1.4.** So the correction codex forced runs *in the
architecture's favour*: it is consistently positive under both statistics rather
than helping under one and hurting under the other, and my "approximately a
no-op in both directions" was wrong in the direction that flattered my own
headline. I have retracted claims against this architecture four times this
weekend and this is the first retraction that helps it, which is worth noting
because the asymmetry would otherwise look like a pattern rather than the
evidence.

What did not change is the size: both deltas are inside one sd of the replicate
spread (15.2 points), and the architecture costs **7.2x the runtime** to buy
+1.4 points (2.0 to 14.6 min). *Small, real-looking, and not worth its price* is
the fair summary.

### Two numbers I typed from estimate, and the guardrail that missed them

Filling WEEKEND.md's `select_k = 40` alpha ladder I typed the permutation arm's
suspect counts as 0.79 and 0.40. The runs say **1.18** and **0.22**.
`audit_weekend.py` passed anyway, because it had no claim covering those two
cells — the audit only checks numbers someone remembered to register, so its
green light means "every registered number traces", not "every number traces".
I caught these by reading the JSON before believing my own table.

The audit now registers both full alpha ladders for both depths, 68 claims. But
the general lesson is about the guardrail rather than the typo: **a
registration-based audit cannot tell you about the number you forgot to
register.** Its silence on a cell is not evidence. The two mitigations that
actually scale are generating the table instead of typing it — which is why
`RESULTS.md` has never had this class of error — and treating any hand-written
table as unverified until each cell has been read back from a JSON.

---

## Turn 11, fourth part — H7 confirmed, and the AUC deficit decomposes

`scripts/eval_attr_arm.py --jobs 16`, one arm under the published protocol,
1.8 min → `runs/attr_arm.json`.

**Prediction, registered before the run:** near `univar_top25_rf`'s 0.730,
inside [0.720, 0.745], paired delta against `rf_all` still negative with a CI
excluding zero.

**Measured: 0.729 [0.710, 0.748].** Inside the interval, 0.001 from the named
landmark.

| paired against | its AUC | `agent_model_rf` minus it | Wilcoxon p |
|---|---|---|---|
| `rf_all` | 0.759 | **-0.0303 [-0.0474, -0.0132]** | 0.002 |
| `univar_top25_rf` | 0.730 | -0.0009 [-0.0146, +0.0129] | 0.711 |
| `agent_rf` | 0.717 | +0.0116 [-0.0006, +0.0237] | 0.071 |

The deficit against the full-sensor forest goes -0.042 to **-0.030**, interval
still excluding zero. And the arm lands exactly on the naive-selection control:
p = 0.71, an interval centred on -0.001. With a decent attribution statistic,
the whole plan/attribute/correlate/verify apparatus performs
*indistinguishably from ranking each sensor on its own and keeping the top 25.*

**The decomposition.** The loop's -0.042 is roughly +0.012 of recoverable
ranking quality plus -0.030 of irreducible sparsity price. Better attribution
collects the first and cannot touch the second, because the second is what any
method paying for a 25-sensor budget owes on a dataset whose signal is spread
across hundreds of weak sensors. **The accuracy deficit is not fixable by any
attribution statistic.**

I registered in advance that this is the *worse* outcome for the architecture,
and I want to be clear that it still is. Under the competing explanation the
deficit was bad ranking, and bad ranking is repairable. Under the confirmed one
the binding constraint is the decision to select at all, which is the
architecture's premise rather than a parameter in it. The three axes now read:
the statistic is worth +13.0 stability points, +2.2 error-control points at
narrow depth (-3.6 at the pre-registered one), and about +0.012 AUC that does
not clear significance. Nothing recovers the accuracy gap.

### Two reasons I am not banking the +0.012, both cutting against my own arm

First, `p = 0.071` and the interval touches zero at -0.0006. Not established.

Second, and I nearly missed this: the model arm selects **25.0** sensors per
fold against the permutation loop's **19.8**, and 25.0 *is* `max_select`
exactly. The arm is pinned against its own budget and would have taken more if
allowed. Selection costs AUC monotonically here, so part of that +0.012 is
simply less sparsity rather than better ranking, and the two arms are not
sparsity-matched. My registered distrust check was aimed at growth toward all
474 sensors, which did not happen — but it was aimed one order of magnitude too
coarsely, and the interesting violation was sitting at the cap.

Both caveats shrink the ranking-quality share and so make the sparsity
explanation stronger. That is a case where every correction happens to favour
the conclusion I predicted, which is exactly when I should say so out loud
rather than let it pass: a sparsity-matched rerun (`max_select` raised for both
arms, or lowered to 19 for the model arm) is the ablation this result now
demands, and until it exists the +0.012 should be read as an upper bound on what
attribution buys, not an estimate of it.

### `PredictAllReportFew`: the correction was worth -0.002 AUC, and that is the finding

`runs/par_few.json`, 25 folds, same protocol, 11.0 min at one worker.

| arm | AUC |
|---|---|
| `par_untuned` (the old default, `min_samples_leaf=5` fixed) | 0.759 [0.740, 0.778] |
| `par_tuned` (the new default, tuned by inner CV) | paired **-0.0021 [-0.0089, +0.0046]** |

Tuning buys nothing measurable, and the JSON says why: across 25 folds the
inner CV picked `min_samples_leaf` = 1 six times, 5 nine times and 10 ten
times. It is genuinely indifferent between them, so the parameter the docstring's
invalid argument skipped over turns out not to matter on this dataset.

So: **the claim was unearned and correct.** The number it asserted (0.759) is
what the arm measures; the argument for it was invalid; and the fix is worth
-0.002 AUC with an interval covering zero. I am reporting the size of my own
correction faithfully because the temptation in a log like this is to let a
found-and-fixed inconsistency read as a save, and this one moved nothing. What
it did buy is that the equality is now true by construction and asserted by a
test, so the next person to retune either side gets a failure instead of a
stale docstring.

---

## Turn 12 (2026-09-04) — H8, written before the run, and evidence I already had and failed to use

Nothing is running; everything queued last turn landed. So this turn opens with
the ablation I registered in `WEEKEND.md` as the one the last result demands:
**the sparsity-matched version of H7.**

### The confound, restated

`agent_model_rf` scores 0.729 against `agent_rf`'s 0.717, a paired
+0.0116 [-0.0006, +0.0237], p = 0.071. But it selects **25.0** sensors per fold
against 19.8, and 25.0 is `max_select` exactly — it is pinned at its budget and
would have taken more. Since selection costs AUC monotonically on this dataset,
some unknown share of that +0.0116 is simply less sparsity. I wrote in
`RESULTS.md` and `WEEKEND.md` that the number should therefore be read as an
upper bound on what attribution buys.

### Evidence already in the repository that I did not use when I wrote that

Reading `runs/secom_loop_sweep.json` before designing the run — which I should
have done before writing the caveat — the sweep contains a relevant dominance
relation that needs no interpolation at all:

| sweep arm | n_selected | AUC |
|---|---|---|
| `agent_wide_permutation` | 32.8 | 0.7364 |
| `agent_operating_model` | **25.0** | **0.7397** |

The model-native arm scores **higher on 8 fewer sensors**. If the attribution
gain were purely a sparsity artifact, that ordering could not happen. So the
existing evidence points the *other* way from my caveat — the effect looks like
it survives sparsity adjustment.

I should be plain that this is a case of my own hedge being less well supported
than the claim it hedged. The caveat was correct that the H7 comparison is
confounded; it was careless in implying the confound probably explains the
effect, when a run already in `runs/` suggested it does not. Two reasons the
sweep does not settle it, which is why the run below is still worth doing: it
uses a 10-fold protocol rather than the 25-fold headline one, and its tags vary
`select_k` and `stability_min` alongside `max_select`, so its arms differ in
more than sparsity and attribution.

### H8

> **H8.** The attribution effect on AUC is not a sparsity artifact. Sweeping
> `max_select` over {5, 10, 15, 20, 25, 40} for both attribution statistics
> under the headline 25-fold protocol — everything else at the pre-registered
> operating point — the model-native curve will lie **above** the permutation
> curve at matched selected-set size, at **every rung where both arms are
> pinned at the cap**, with the matched-*n* difference in the range
> **+0.005 to +0.020** rather than collapsing into ±0.005 of zero.

**What distinguishes H8 from the obvious alternative.** The alternative is the
one my own caveat asserted: the +0.0116 is mostly the extra 5.2 sensors, so at
matched *n* the curves coincide. It predicts matched-*n* differences scattered
around zero and a sign pattern no better than chance across the ladder.

**How it will be judged, decided now rather than after seeing it.** A single
rung cannot resolve this: the paired CI half-width on this protocol is about
0.012, wider than the gap between the two predictions. So the evidence is the
**sign pattern across matched rungs**, reported as a count (as this repo did for
the 5-of-5 depth comparison) and not as a pooled CI — the rungs share folds and
nest, so pooling them would manufacture precision I have not earned. Individual
per-rung paired CIs go in the table and most of them will straddle zero; that is
expected and is not a result either way.

**Second thing this buys, which may matter more than H8 itself.** The ladder
measures dAUC/d(n_selected) for each statistic *under the headline protocol*.
The "-0.030 of irreducible sparsity price" in the H7 write-up currently rests on
a comparison with `univar_top25_rf` at one budget plus a monotonicity claim
imported from the coarser 10-fold sweep. A curve on the headline protocol either
supports that number or replaces it.

**Third, and registered as a check on myself:** if the model curve turns out to
lie *below* the permutation curve at matched *n*, then the entire +0.0116 was
sparsity, my caveat was right, and the H7 write-up's "recoverable ranking
quality" term goes to zero — which would make the sparsity explanation of the
loop's accuracy deficit *total* rather than merely dominant. That is a cleaner
result than H8 and I would rather have it than be right.

**Run:** `scripts/eval_sparsity.py --jobs 16`, 12 arms x 25 folds, paired on
identical folds, ~30 min (permutation fits are 127 s against the model arm's
~35 s). Nothing already published is recomputed.

---

## Turn 12, second part — codex takes the ceiling apart, and is right about all of it

Asked `codex exec` for the strongest reason the saturation ceiling is "wrong,
circular, trivial, or mis-stated", naming three specific angles. It found
problems on all three plus a scope error, and I am accepting every one.

> The strongest objection is that this is a tautological endpoint bound, not an
> identity and not a substantive property of max-support calibration. [...]
> Because the fitted quantile satisfies tau <= 1, P(M < tau) <= P(M < 1) =
> 1 - P(M = 1). That is the entire "ceiling." It follows immediately from
> boundedness and the `>=` comparison [...] It does not explain why saturation
> occurs, predict its probability, or distinguish this procedure from any other
> bounded score.

Correct, and it is the objection I should have anticipated. I called it an
"identity" that "predicts rather than describes". It is an inequality, it is
one line from boundedness plus the comparison operator, and calling it a
discovery was self-flattery.

> The "independent of alpha" language is misleading. The endpoint bound is
> numerically independent of alpha, but whether it binds is highly
> alpha-dependent because alpha determines whether the quantile reaches 1.

Correct.

> the report/test classify a level as pinned using only `tau_mean >= 0.999`,
> which does not establish that every split actually used tau = 1.

Correct, and **checkably** so — I had `tau_min` in the JSON and used the mean.
`abstain_model` at alpha = 0.05 has `tau_mean` = 0.9999 but `tau_min` = 0.9208,
so my "pinned" label on that row was false by the criterion I claimed to be
using.

> P(M=1) is not a property of "the threshold rule". It depends on the bootstrap
> count, selection depth, attribution/ranking method, candidate count, tie
> handling, null-generation scheme [...] It is therefore a property of the
> complete score-generating experiment.

Correct. So is the scope error it adds: a rule comparing with strict `>` at 1,
or thresholding an unbounded score, escapes the bound entirely, which makes
"any max-support threshold rule" false as written.

### What chasing the `tau_min` point turned up, which is the actual result

Following codex's (b) to its end produced something better than the thing it
demolished. If `tau_min` = 0.9208 at alpha = 0.05 and `tau_min` = 1.0 at
alpha = 0.01, why is the measured control **0.880000 at both, to six decimals**?

Because the statistic is discrete. Support is a count over `n_boot` = 12
resamples, so the null max lives on multiples of 1/12. The observed null max
values are exactly {1.000 (24 replicates), 0.9167 (47), 0.8333 (54), 0.750
(46), ...} — **nothing between 11/12 and 1**. Every threshold in that gap
selects the same replicates and returns the same number. Alpha moves the
threshold within a gap without moving the answer.

Which generalises: control as a function of tau is a **step function**, and only
`n_boot` + 1 values are attainable at any alpha:

> attainable control = { 1 - P(M >= k / `n_boot`) : k = 1 .. `n_boot` }

| arm | P(M=1) | attainable control above 0.60 | closest to 0.95 |
|---|---|---|---|
| `select_k = 40`, permutation | 1.5% | 0.985, 0.920, 0.790 | 0.920 |
| `select_k = 40`, model | 12.0% | 0.880, 0.645 | 0.880 |
| `select_k = 5`, permutation | 0.0% | 1.000, 0.995, 0.955, 0.875, 0.625 | 0.955 |
| `select_k = 5`, model | 0.5% | 0.995, 0.990, 0.965, 0.935, 0.905, 0.805, 0.655 | 0.935 |

**No arm can land on 0.95, and for one of them the entire attainable set above
0.60 is a single value.** That is not a calibration failure, it is a resolution
limit, and the parameter that sets it is `n_boot` — chosen in this repo for
compute cost, never once discussed as governing the false-discovery guarantee.
The endpoint bound codex called trivial is one corner of this; the coarseness
is the part worth reporting, and it is not a restatement of how quantiles work.

Rewritten in `RESULTS.md` under "What error control is achievable at all", with
the trivial corner labelled trivial in the text, the alpha language corrected,
the scope restricted to these arms under this protocol, and `tau_min`/`tau_max`
reported per level. The test is replaced too: it no longer asserts a "cap
identity" but checks that measured control sits on the grid and that every
exact tie between alpha levels is explained by a gap containing no null
replicate. `paper_draft.md` 5.6 is rewritten and no longer calls this the most
transferable finding in the paper.

---

## Turn 12, third part — H8 refuted: the AUC effect was sparsity all along

`scripts/eval_sparsity.py --jobs 16`, 300 fits, 22.1 min → `runs/sparsity.json`.
`max_select` swept over {5, 10, 15, 20, 25, 40} for both attribution
statistics, everything else at the pre-registered operating point, headline
25-fold protocol, paired per fold.

| `max_select` | perm AUC | perm n | model AUC | model n | n gap | model - perm |
|---|---|---|---|---|---|---|
| 5 | 0.6865 | 5.0 | 0.6843 | 5.0 | +0.0 | **-0.0022** |
| 10 | 0.7045 | 9.9 | 0.7015 | 10.0 | +0.1 | **-0.0031** |
| 15 | 0.7146 | 14.5 | 0.7193 | 15.0 | +0.5 | +0.0046 |
| 20 | 0.7193 | 17.9 | 0.7256 | 20.0 | +2.1 | +0.0064 |
| 25 | 0.7172 | 19.8 | 0.7288 | 25.0 | +5.1 | +0.0116 |
| 40 | 0.7199 | 21.8 | 0.7368 | 32.1 | +10.3 | +0.0168 |

**H8 predicted the model curve would sit above the permutation curve at matched
size by +0.005 to +0.020. At the two rungs where the arms take the same number
of sensors it sits fractionally *below*, at -0.0022 and -0.0031.** The
pre-registered sign count is 0 of 1 strictly matched rungs, and the fuller
picture is stronger than that count: the paired difference correlates with the
sparsity gap at **r = 0.924**, and at 0.00284 AUC per sensor the 5.1-sensor gap
at `max_select` = 25 predicts +0.0145 against the +0.0116 observed. Sparsity
alone accounts for the effect.

So the attribution statistic buys **nothing** on accuracy. It is worth 13 points
of selection stability, 2.2 points of error control at a suitable depth, and
zero AUC.

### Which of my own two statements was right, and why the wrong one was wrong

This repository said both of these about the same number, one turn apart:

1. (Turn 11, in `RESULTS.md`) read +0.012 as an upper bound, not an estimate,
   because the arms are not sparsity-matched.
2. (Turn 12, pre-run) the effect will survive matching, because
   `runs/secom_loop_sweep.json` shows the model arm scoring higher on 8 fewer
   sensors.

(1) was right. (2) was wrong, and the reason is instructive: those sweep arms
differ in `select_k` and `stability_min` as well as in sparsity and attribution.
**I wrote that caveat down in the same paragraph and then let the dominance
relation drive the prediction anyway.** Noting a confound is not the same as
propagating it, and I did the first without the second.

I registered before the run that a refutation here would be the cleaner result
and that I would rather have it than be right. That was easy to write and I want
to check it against what actually happened: the refutation removes a +0.012 term
I had already published a decomposition around, so the H7 write-up needed a
withdrawal rather than a footnote. It is withdrawn in `RESULTS.md` in place, and
the headline conclusion it supported is not weakened but strengthened — the
loop's -0.042 AUC deficit is **sparsity price in full**, with no recoverable
ranking component at all.

`n_boot` = 12 and `max_select` are now the two parameters this repository can
show govern its headline metrics, and neither was ever tuned or discussed.

---

## Turn 12, fourth part — H9, written before the run

The grid finding makes a prediction, which is the test of whether it is worth
anything. Error control is a step function on {1 - P(M >= k/`n_boot`)}, so
`n_boot` sets the resolution. At `n_boot` = 12 the closest attainable value to
0.95 for the best arm (`select_k` = 5, model attribution) is **0.935**.

> **H9.** Raising `n_boot` from 12 to 40, changing nothing else, refines the
> attainable set enough to bring a value within **0.01 of 0.95**, and the
> measured held-out control at alpha = 0.05 lands closer to nominal than the
> current 93.7%.

**What distinguishes H9 from the obvious alternative, and why I think it is
close to even money.** Two effects run in opposite directions and the repo has
measured neither:

1. *Spacing.* The grid goes from steps of 1/12 to 1/40, so more values become
   attainable. This helps.
2. *Saturation.* P(M = 1) may **rise**. A sensor selected in 12 of 12 resamples
   is a weaker claim than one selected in 40 of 40, but it is not obvious which
   way the mass moves: more resamples give a noisy sensor more chances to miss,
   which lowers P(M = 1), while the extra resamples also average out the noise
   that was making it miss, which raises it. If saturation rises, the top of
   the attainable set drops and the finer spacing buys nothing.

If (1) dominates, control moves toward 0.95 and H9 holds. If (2) dominates,
control gets *worse* despite the finer grid, and the practical advice inverts:
fewer bootstrap resamples, not more. Either outcome is publishable and the
second is more interesting, since "raise `n_boot`" is what anyone would guess.

**A third possibility I am registering now so it is not a post-hoc save:** the
grid may refine while the *calibration* fails to exploit it, because tau is
fitted on 100 null replicates per half-split and a finer grid needs more
replicates to resolve. If the attainable set contains a value near 0.95 but the
measured control does not move toward it, that is the diagnosis, and the check
is whether `tau_min` and `tau_max` spread further apart across splits than they
do at `n_boot` = 12.

**Run:** `scripts/null_fdr.py --null 200 --real 40 --jobs 16 --base rf
--select-k 5 --attribution model --n-boot 40`, priced by the same
`scripts/abstain.py` split-half calibration →
`runs/null_fdr_k5_model_b40.json`, `runs/abstain_k5_model_b40.json`. About 3.3x
the bootstrap work of the `n_boot` = 12 arm, so roughly an hour. The
`--n-boot` flag is a two-line addition of the same shape as `--select-k` and
`--attribution`; nothing already measured is recomputed.
