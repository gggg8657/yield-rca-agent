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
