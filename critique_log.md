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
