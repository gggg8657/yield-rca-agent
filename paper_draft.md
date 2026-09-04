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
false-discovery rate of 1.0 under a label-permutation null: it reports root
causes on every replicate of a dataset in which no sensor carries any
information about failure, and its confidence scores on real labels are
[separation result] from its confidence scores on noise. We trace this to an
architectural property rather than a hyperparameter, propose a null-calibrated
abstention rule with family-wise error control, and measure what enforcing it
costs. Third, we ask whether the surviving suspects support a causal reading via
an invariance screen across production periods, and report a negative result
with its power curve attached: at 104 failed wafers the screen cannot detect a
broken association smaller than [power result], so the pipeline's output is
associational and we say so. Finally we contrast all of this against a synthetic
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
3. **An invariance screen across production periods, reported with its power
   curve**, so that a null result is interpretable rather than flattering. (§6)
4. **A negative accuracy result against the obvious baseline**, and the
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
none?" from `runs/null_fdr.json`.

The result to write up here has three parts: the abstention rate on the null,
the false-discovery rate given a non-empty report, and
P(real replicate's best support > null replicate's best support) as the
common-language measure of whether the loop's own confidence knows the
difference.

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

`[not measured]` -- this is the change queued for the next run: implement
abstention as a pipeline option, then measure (a) that the null abstention rate
rises to the intended `1 - alpha`, and (b) what fraction of real-label
replicates return an empty report, which is the price of the guarantee.

The honest framing is that this converts an unfalsifiable claim ("unstable
suspects are dropped") into a falsifiable one with a knob, and that the knob's
setting is a business decision about the cost of a wasted engineering
investigation versus a missed cause.

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

**Anticipated conclusion:** the pipeline reports associational suspects, the
data cannot support upgrading that, and the repo says so wherever suspects are
named. This is a better outcome than an unearned causal claim.

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
2. The agents are deterministic tool users, not language models. This buys
   reproducibility and means the hallucination measured here is *structural*
   rather than generative -- an LLM narrator over the same tool outputs would
   add a second, unmeasured failure mode on top.
3. The invariance screen is marginal, not full ICP, and underpowered at this
   sample size -- quantified rather than glossed (§6.3).
4. 104 failures bounds everything. Several results here are "this dataset cannot
   settle it", which is honest but is not the same as settled.

---

## 10. What a reader should take away

Report a false-discovery rate under a null, or do not claim your agent drops bad
suspects. It costs one permutation loop, it is the only version of the claim
that can be wrong, and on the pipeline studied here it changes the conclusion
from "verified suspects" to "a procedure that cannot abstain".

---

## Appendix: reproduction

`bash scripts/overnight.sh ~/miniforge3/envs/pybamm-inv/bin/python` regenerates
every JSON under `runs/` in dependency order, then `scripts/report.py --check`
fails if any document disagrees with any JSON. CPU only, 16 workers.
