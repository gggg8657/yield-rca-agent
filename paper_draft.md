# When a root-cause agent has nothing to say: false-discovery control for multi-agent attribution on semiconductor yield data

*Working draft. Every number in this draft is either already generated into
`RESULTS.md` by `scripts/report.py` from a JSON under `runs/`, or marked
`[not measured]`. Nothing is typed in from memory, and no number from a
published paper appears except as an explicitly labelled published baseline.*

---

## Abstract (draft)

Multi-agent pipelines for industrial root-cause analysis are typically evaluated
on their prediction accuracy and on the plausibility of the causes they name. We
argue both are the wrong test, and demonstrate a third on UCI SECOM. First, on
this dataset a plain random forest over all sensors beats the full plan /
attribute / verify / drop agent loop by a margin whose paired 95% interval
excludes zero, so the agent architecture costs accuracy rather than buying it.
Second, and more seriously, we show that the loop's advertised safeguard --
suspects failing a bootstrap stability check are dropped -- has a
false-discovery rate of 1.0 under a label-permutation null: across 200
replicates of a dataset in which no sensor carries any information about
failure, it named 2,743 root causes and abstained on none. The cause is not the
never-return-empty-handed guard the code invites one to blame -- that fired on
0.0% of replicates -- but a stability threshold set an order of magnitude below
what noise clears. The underlying statistic is nonetheless informative
(P(real > null) = 0.873), so we derive a null-calibrated abstention rule with
family-wise control and price it under held-out calibration: at alpha = 0.05 the
honest report is 0.60 suspects and empty 51% of the time, against the 20.9 the
pipeline prints. Third, we ask whether anything simpler would have
calibrated as well, and find that it does the whole job better: a univariate
ranker at a selection depth where its support does not saturate separates the
two worlds at 1.000, reaches 94.3% error control against the loop's 91.6%, and
reports 2.06 suspects against 0.60 -- with no permutation-importance pass, no
correlation grouping and no verification loop. Fourth, we ask whether the
surviving suspects support a causal reading via an invariance screen across
production periods. Exactly one of 22
associated sensors is rejected -- and it is the one the loop ranks first in
25 of 25 folds -- while the remaining 21 are not invariant but untestable, which
we establish by attaching the screen's power curve: at 104 failed wafers it
cannot see a break smaller than roughly 0.15 AUC, and the sensors' entire signal
is smaller than that. The pipeline's output is therefore associational and we
say so. Taken together the agent loop has no measured advantage over a
univariate ranker on any axis we evaluate; it wins only on a synthetic generator
where its premise -- that a few sensors dominate -- holds by construction, which
localises the failure in the premise rather than the implementation. Finally we contrast all of this against a synthetic
generator with known causes, where the same loop wins on both accuracy and
stability -- locating the failure precisely in the premise that a few sensors
dominate, rather than in the implementation.

---

## 1. Introduction

### The claim structure that industrial RCA papers use, and why it is untestable

A yield root-cause tool makes two claims that are usually evaluated together and
should not be:

1. *Prediction.* Given process sensors, it can flag which wafers will fail.
2. *Attribution.* Of the hundreds of sensors, these five are the ones to go look
   at.

Claim 1 has a standard evaluation. Claim 2 does not, on real data, because real
fab datasets do not come with ground-truth causes -- and the datasets that do
are simulated. The field's usual substitutes are (a) a domain expert finding the
named sensors plausible, and (b) a stability or agreement statistic over
resamples. Neither is a false-discovery rate, and neither can tell you what the
tool does when there is nothing to find.

That gap matters most for exactly the systems now being built on top of LLM
agents, because their failure mode is fluency: a report naming five sensors,
with impact scores and a confidence percentage, reads identically whether or not
the underlying data contains a signal. An engineer cannot tell the two apart
from the artefact.

### What this paper does

We take a deterministic multi-agent RCA pipeline -- no language model, so every
number is reproducible from a seed -- and subject its *report* to the test its
*predictor* already gets. The pipeline is deliberately conventional: a screening
agent, a permutation-importance attribution agent, a correlation-grouping agent,
a bootstrap verification agent that drops unstable suspects, and a reporting
agent. This is the architecture the literature describes, implemented honestly,
and evaluated with everything fitted inside the cross-validation fold.

Contributions:

1. **A false-discovery measurement for agent attribution.** Permute the labels,
   run the entire loop, count the causes it invents. (§4)
2. **A null-calibrated abstention rule** derived from that null, giving the
   pipeline the ability -- which it structurally lacks -- to report nothing. (§5)
3. **A saturation diagnostic for calibrated selection.** Bootstrap-support
   thresholds are widely used to filter unstable features; we show the statistic
   can pin at its ceiling on a non-trivial share of *null* replicates, capping
   the error control any threshold can deliver, and that this is a property of
   the selection depth rather than of the ranker. Reporting the attainable
   ceiling alongside the achieved rate is a one-line change that prevents the
   conclusion we ourselves drew and had to retract. (§5.3)
4. **An invariance screen across production periods, reported with its power
   curve**, so that a null result is interpretable rather than flattering. (§6)
5. **A negative accuracy result against the obvious baseline**, and the
   diagnosis that separates "the loop is badly implemented" from "the loop's
   premise does not hold on this data". (§3, §7)

---

## 2. Setup

### 2.1 Data

UCI SECOM: 1,567 wafers, 590 sensors, 104 failures. Full descriptive profile in
`RESULTS.md` §"The data, as it actually arrives", generated from
`runs/data_profile.json`.

Two facts about it drive most of what follows and are measured in
`runs/drift.json`: the sensor matrix predicts *when* a wafer was made at
adversarial AUC 0.993, and the failure rate falls monotonically across time
blocks. SECOM is not one distribution.

### 2.2 The two worlds, kept apart

| | ground-truth causes | what can be claimed |
|---|---|---|
| UCI SECOM (real) | none exist | prediction, selection stability, false-discovery rate under a null, invariance |
| `make_synthetic` (simulated) | 5 of 200 by construction | all of the above **plus** recovery of the true causal set |

No recovery claim is made for SECOM anywhere in this work. Every synthetic
number is labelled simulated at the point of use. This separation is enforced in
the artefact, not just in the prose: the synthetic results live in their own
section of `RESULTS.md` and their own JSON.

### 2.3 Protocol

RepeatedStratifiedKFold, 5 splits x 5 repeats, byte-identical folds for every
arm, paired comparisons over the 25 folds. Cleaning, imputation, standardisation,
missing-indicator selection, candidate screening, permutation importance,
bootstrap verification and baseline hyperparameter search are all fitted inside
the training fold. Baselines get an inner 3-fold grid search; the agent loop's
structural settings are pre-registered at one operating point and the full
sensitivity surface is published rather than tuned.

Because the shuffled-CV protocol mixes production eras, we also report a
chronological split and a rolling-origin (expanding-window) protocol. §7 argues
these, not the shuffled numbers, are the deployment-relevant ones.

---

## 3. The agent loop loses to a random forest

Numbers: `RESULTS.md` §"SECOM: prediction", from `runs/secom_eval.json`.

The headline is that the best plain baseline beats both agent arms by a paired
margin whose 95% interval excludes zero over 25 folds. At *matched sparsity* --
25 sensors each -- the loop's advantage over picking 25 sensors one at a time is
an interval that includes zero. The comparison the architecture most needs to
win, it does not win.

The sensitivity sweep (`runs/secom_loop_sweep.json`) shows held-out AUC is
monotone in how many sensors survive, and the loop only matches the baseline
once it stops being a shortlist. Disabling the drop step returns it to baseline,
which rules out a wrapper bug as the explanation.

**Diagnosis.** SECOM's signal is diffuse. Enforcing sparsity discards
information the classifier was using. §7 contrasts this with the synthetic
generator, where the sparsity premise holds and the same loop wins.

---

## 4. Does the loop invent root causes? (the central experiment)

### 4.1 The null

Permute the labels over all wafers, preserving the 104/1,463 class balance
exactly, and leave the sensor matrix untouched. No sensor carries information
about failure by construction, so **every sensor the pipeline names on such a
replicate is a false discovery** -- not "probably false", false. Run the entire
agent loop, unchanged, on many such replicates.

The comparison arm is the identical loop on the true labels with its internal
randomness re-seeded, so that both sides are distributions over the same
statistics rather than a distribution against a point.

### 4.2 What is measured

Per replicate: how many sensors are reported; how many cleared the stability
threshold on their own merit; whether the loop's never-return-empty-handed
fallback fired; and the largest bootstrap support any suspect achieved -- the
statistic the drop step thresholds, and therefore the one whose null
distribution decides whether that threshold is a decision or a formality.

### 4.3 Result

Generated into `RESULTS.md` §"Does the loop invent root causes when there are
none?" from `runs/null_fdr.json`; 200 permuted and 40 real replicates, 31.5 min
on 16 CPU workers.

| | permuted labels | real labels |
|---|---|---|
| suspects reported per replicate | 13.7 | 20.9 |
| cleared the threshold on merit | 13.7 | 21.1 |
| abstention rate | 0.0% | 0.0% |
| never-empty fallback fired | 0.0% | 0.0% |
| largest bootstrap support | 0.703 | 0.873 |

**2,743 false discoveries over 200 replicates; FDR of the reported list = 1.0;
abstention rate 0.** The loop never once declined to name a cause on data with
no causes in it.

Two of our own predictions failed, and both failures are informative. We
expected the never-empty guards to be the mechanism; they fired on 0.0% of
replicates and were never needed, because `stability_min = 0.3` is itself far
below the noise floor -- a sensor need only reach the top 40 of a 60-sensor pool
in 4 of 12 bootstrap replicates. And we expected the null and real support
distributions to overlap; they separate at P(real > null) = 0.873
(Mann-Whitney p = 1.6e-14). The second failure is what makes §5 worth writing:
had the statistic been uninformative, no threshold could have rescued it and the
attribution step itself would have needed replacing.

**The null is not unfairly easy.** The competing explanation is that permuting
labels leaves the sensor correlation structure intact, so the loop reports that
structure rather than inventing. Then null replicates would agree with each
other. They agree on 0.014 of their top-5 against a random-ranker floor of
0.011, having named 417 of 474 sensors at least once across the run. The
invented causes are fresh each time.

### 4.4 Why this is architecture, not tuning

`AgentRCA.fit` contains two guards that make an empty report impossible: if no
candidate has positive importance it takes the least-bad candidates, and if no
suspect clears the stability threshold it restores the top five. An independent
review by a second CLI agent found the same lines unprompted (quoted in
`critique_log.md`). Lowering or raising `stability_min` cannot fix this, because
the guard fires precisely when the threshold would have filtered everything --
the situation in which withholding a report is the only correct action.

---

## 5. Giving the pipeline the ability to say nothing

### 5.1 The rule

Let `s_j` be suspect *j*'s bootstrap support and let `tau(alpha)` be the
`(1 - alpha)` quantile of the *maximum* support observed across null replicates.
Reporting only suspects with `s_j >= tau(alpha)` is Westfall-Young
max-statistic control: family-wise over the sensors screened in a replicate,
distribution-free, and calibrated on the pipeline's own behaviour rather than on
an asymptotic argument that its bootstrap does not satisfy.

### 5.2 What it costs

Calibration is held out: fitting `tau` and measuring abstention on the same
replicates returns `1 - alpha` by construction, so the null replicates are
halved, `tau` is fitted on one half and every rate read off the other, averaged
over both directions and 400 random partitions. No model is refitted --
everything is a function of the supports already recorded.

| alpha | tau | held-out null silent | SECOM silent | SECOM suspects |
|---|---|---|---|---|
| 0.10 | 0.842 | 82.0% (target 90%) | 23.4% | 1.18 |
| 0.05 | 0.910 | 91.6% (target 95%) | 51.3% | 0.60 |
| 0.01 | 0.959 | 97.7% (target 99%) | 78.8% | 0.22 |
| none | -- | 0.0% | 0.0% | 20.95 |

Two imperfections in the rule, both reported rather than smoothed. It
**under-abstains** -- 91.6% against a nominal 95% -- because `tau` is a point
estimate of an upper quantile from 100 replicates and such estimates are biased
low; the remedy is more null replicates or an upper confidence bound on the
quantile. And the bar sits on a **13-point grid**, since support is a fraction
of `n_boot = 12`, which is why alpha = 0.01 buys little over alpha = 0.05.

This converts an unfalsifiable claim ("unstable suspects are dropped") into a
falsifiable one with a knob, and moves the deliverable from "your five root
causes" to "at most one or two, often none, and the reason why". Where alpha
should sit is a business question about the cost of a wasted investigation
against a missed cause, not a statistical one.

### 5.3 Would anything simpler have calibrated as well?

Comparing raw false-discovery rates across methods is uninformative: any
procedure that always emits a top-k has FDR 1.0 under this null. Two properties
do discriminate, and they can disagree -- how well the statistic *separates* a
world with causes from one without, and how much error control it can be
*thresholded* to. We compare the loop against univariate, logistic-coefficient
and impurity rankers, every arm matched to the loop's own bootstrap count and
selection depth and scored with the identical max statistic and the identical
held-out calibration.

At matched settings the plain rankers separate better (univariate 0.943 vs
0.873) but their support **saturates**: their best sensor sits in the top slice
of every resample, so the statistic pins at 1.000 on real labels and on 11.5% of
permuted ones. No threshold at or below 1.000 excludes those, capping univariate
at 88.5% control against the loop's 91.6%. Read alone, this says the loop's
noisier estimator is the only one able to carry a guarantee -- a conclusion we
drew, wrote up, and then refuted.

Narrowing the selection depth removes the saturation entirely. At
`select_k = 5`, univariate reaches a 100% ceiling, 94.3% achieved control,
1.000 separation and 2.06 reported suspects against the loop's 0.60 and 51%
abstention. **What saturated was the depth, not the ranker.** The methodological
point generalises beyond this pipeline: any stability-selection procedure that
thresholds a bounded support statistic should report the attainable ceiling next
to the achieved rate, because a saturated statistic looks maximally confident
and is minimally calibratable.

### 5.4 Ranker or depth? The symmetric comparison

Tuning the baseline's depth while leaving the pipeline at its pre-registered one
answers "would something simpler have sufficed" and not "is the architecture
worse at equal effort". We therefore ran the loop at the same `select_k = 5`,
changing nothing else.

Its error control does not move: 91.6% to 91.5%. Depth explains the baseline's
behaviour entirely and the loop's not at all, which localises the loop's
disadvantage in the held-out permutation-importance estimator rather than in a
tunable parameter. At matched depth the univariate ranker still leads control by
2.8 points and reports 2.06 suspects against 1.16, so the equal-effort
comparison reaches the same conclusion as the tuned one.

The unanticipated finding is more interesting than the predicted one. On
permuted labels the two guards inside `fit` behave oppositely at the two depths:

| on permuted labels | `select_k = 40` | `select_k = 5` |
|---|---|---|
| noise sensors clearing the threshold on merit | 13.7 | 0.47 |
| never-empty fallback fired | 0.0% | 62.5% |
| replicates reporting nothing | 0.0% | 0.0% |

At the loose depth the stability threshold filters nothing and the fallback is
never invoked. At the tight depth **the threshold works almost perfectly and the
fallback then reinstates the sensors it removed**, on nearly two thirds of null
replicates. The false-discovery rate is 1.0 at both depths, by two mechanisms
that share nothing.

This is worth stating as a general caution about ablating agent pipelines. A
component that appears inert at one operating point — the fallback fired on 0%
of replicates and looked like dead code — can be the decisive one at another. We
drew the inert reading first, wrote it up, and had to retract it when the second
depth was measured. Ablations of such systems should report the operating point
they were run at, and a component should not be called redundant on the strength
of a single one.

---

## 6. Are the suspects causal? A negative result with its power attached

### 6.1 Why invariance, and what it can and cannot establish

Permutation importance measures a model's reliance on a column. To say anything
causal one needs an identification argument. We use the marginal screen from
Invariant Causal Prediction (Peters, Bühlmann & Meinshausen, JRSS-B 2016):
across environments in which the mechanism is unchanged but the covariate
distribution moves, a genuine cause keeps its relationship with the response.
SECOM's contiguous time blocks are candidate environments, and `runs/drift.json`
establishes independently that they differ.

This is a **necessary, not sufficient** condition, and a marginal screen rather
than full ICP -- with 474 sensors and 104 failures, subset search is neither
computable nor powered. Passing does not make a sensor causal; failing rules it
out as a stable cause.

### 6.2 Two stages need two different nulls

The mistake available here is to test invariance by permuting labels. That
builds the reference distribution at AUC 0.5, where the statistic has smaller
sampling variance than at the observed association, and would flag strongly
associated sensors as non-invariant *for their strength alone*. We therefore
permute **block membership**, reshuffling wafers into same-sized periods, which
holds each sensor's pooled association fixed and destroys only the block
structure. Association itself is tested by permuting labels, which is the
correct null for that question.

We took the wrong path first and it produced a materially different answer; the
correction and both numbers are recorded in `critique_log.md`. The closed-form
chi-square reference is also anticonservative on this data because SECOM's
sensors carry heavy ties, so all decisions use permutation p-values and the
chi-square figure is retained only as a diagnostic.

### 6.3 Result, and the power curve that makes it readable

Generated into `RESULTS.md` §"Are the suspects causal, or only associated?" from
`runs/invariance.json`.

The essential companion is the power ladder: sensors built with a *known* break
in their association are put through the identical test, so that "nothing was
flagged non-invariant" can be read as "no break larger than X", with X measured.
Reporting an invariance null result without this is how a powerless test gets
written up as evidence of causality.

**Result.** 22 of 474 sensors are associated with failure (BH FDR 0.05, exact
permutation, B = 20,000). Exactly one is rejected as non-invariant:
`sensor_059`, per-period AUCs 0.56 / 0.77 / 0.86 / 0.52 / 0.49, I-squared 0.82,
BH p = 0.012 -- and it is the sensor the loop ranks in its top 5 in **25 of 25**
cross-validation folds. Its association is strong in the middle of the record
and absent at the end, which is the signature of a period-specific artefact and
exactly what `runs/drift.json` predicts is common here.

The other 21 sensors are **untestable, not invariant**, and the power ladder is
what licenses that distinction. Injected sensors carrying a known break are
detected 14% / 23% / 52% / 79% / 95% of the time at first-period AUCs of
0.55 / 0.60 / 0.65 / 0.70 / 0.75; SECOM's associated sensors carry
|AUC - 0.5| of 0.088 to 0.192, median 0.114. Their entire signal is smaller than
the break the screen needs to see. Note the direction: power rises with
association strength, so the sensors the test can adjudicate are precisely the
ones the pipeline is most confident about -- and the single adjudicable one
failed.

**Conclusion.** The pipeline reports associational suspects, the data cannot
upgrade that, and the artefact says so wherever suspects are named. This is a
better outcome than an unearned causal claim, and it is a stronger negative
result than an unpowered null would have been.

---

## 7. Where the machinery does work, and what that localises

Synthetic generator, 5 genuinely causal sensors among 200, block-correlated
noise, 10 independent datasets. **Simulated data; these numbers are not SECOM
results.** Full table in `RESULTS.md` §"Synthetic benchmark".

There the loop beats the full-sensor forest on held-out AUC, recovers the causal
set at high recall and precision, and *meets* the top-5 stability target it
misses on SECOM by a wide margin.

The contrast is the most useful thing in this work. The loop's premise is that a
few sensors dominate the failures. Where that premise holds it wins on accuracy
and on stability; where the signal is spread thin across hundreds of weak,
drifting sensors, enforcing sparsity throws away what the model needed. The
failure is in the premise, not the implementation -- which is a claim about when
to deploy such a pipeline, and it is testable before deployment by measuring
signal concentration rather than by running the agent loop and hoping.

---

## 8. Forward in time

The shuffled-CV numbers mix production eras, and an adversarial validation AUC
of 0.993 says the eras are nearly perfectly separable. A reviewer of this work
noted, correctly, that calling the shuffled-CV figure "leak-free" is too strong:
it avoids preprocessing leakage but still lets future process regimes inform
training, so it estimates retrospective interpolation rather than prospective
prediction (quoted in `critique_log.md`).

Under a chronological split and a rolling-origin protocol every arm's interval
includes chance. Interestingly the *ordering* inverts -- the agent arm becomes
the best forward-in-time arm, at 5 of 5 block counts by sign -- but not one
paired interval excludes zero, and origins within a block count share training
data. The defensible statement is a direction, not an effect size, and SECOM is
too small to settle it.

---

## 9. Limitations

1. Single real dataset. The false-discovery methodology transfers; the numbers
   do not.
2. The ranker comparison scores *ranking statistics under bootstrap
   resampling*, which is not everything the loop does -- correlation grouping
   and the written report are unscored, and no protocol here could score them.
3. The agents are deterministic tool users, not language models. This buys
   reproducibility and means the hallucination measured here is *structural*
   rather than generative -- an LLM narrator over the same tool outputs would
   add a second, unmeasured failure mode on top.
4. The invariance screen is marginal, not full ICP, and underpowered at this
   sample size -- quantified rather than glossed (§6.3).
5. 104 failures bounds everything. Several results here are "this dataset cannot
   settle it", which is honest but is not the same as settled.
6. Two conclusions in this work were drawn from a single operating point and
   refuted by measuring a second: the saturation ceiling (§5.3) and the
   never-empty guard (§5.4). We report both the refuted and the surviving
   reading, and we cannot rule out that other claims here are similarly
   depth-specific -- the operating points we happened to probe are not a
   systematic sweep.

---

## 10. What a reader should take away

Report a false-discovery rate under a null, or do not claim your agent drops bad
suspects. It costs one permutation loop, it is the only version of the claim
that can be wrong, and on the pipeline studied here it changes the conclusion
from "verified suspects" to "a procedure that cannot abstain".

Then compare against the simplest thing that could work, at *its* best operating
point rather than at yours. We did not, for one revision, and concluded that the
agent architecture uniquely supported a false-discovery guarantee. It did not;
the baseline had been pinned at a selection depth that made its statistic
degenerate. The finding survived only because every number in the write-up was
generated from a run artefact by a script, so re-running turned a refuted
sentence into a diff instead of leaving it in the paper.

---

## Appendix: reproduction

`bash scripts/overnight.sh ~/miniforge3/envs/pybamm-inv/bin/python` regenerates
every JSON under `runs/` in dependency order, then `scripts/report.py --check`
fails if any document disagrees with any JSON. CPU only, 16 workers.
