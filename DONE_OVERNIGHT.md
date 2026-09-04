# Overnight run — 2026-09-04

Took `yield-rca-agent` from a synthetic benchmark to a real-data result on UCI
SECOM. Every number below comes from a run executed in this session and is
regenerated into [`RESULTS.md`](RESULTS.md) and the README by
`scripts/report.py` from the JSONs in `runs/`. `scripts/report.py --check`
fails CI if either drifts, so a stale number cannot survive a commit.

---

## 1. The first thing found, before any new measurement

**The repo's headline numbers were in-sample.** `run_rca` fitted on all rows
and scored the same rows: that is where "AUC 0.998, 5/5 recovered" came from.
On the same synthetic generator, 5-fold cross-validated AUC is **0.930**. The
gap is not cosmetic, and on 590 SECOM sensors it would have been much larger.

`yieldrca/pipeline.py` now reports a cross-validated AUC and keeps
`auc_in_sample` beside it, explicitly named, with a test asserting the headline
is the held-out one. A second bug: `roc_auc` had no tie handling, so a
constant-score classifier scored 0.0 or 1.0 depending on input order instead of
0.5. It now matches scikit-learn on tie-heavy inputs.

## 2. What ran

| stage | script | output | wall |
|---|---|---|---|
| dataset profile | `prepare_data.py` | `runs/data_profile.json` | seconds |
| headline evaluation, 7 arms × 25 folds + chronological + rolling-origin | `eval_secom.py` | `runs/secom_eval.json` | 8.5 min CV |
| agent-loop sensitivity, 12 configs × 10 folds | `sweep_loop.py` | `runs/secom_loop_sweep.json` | 12.3 min |
| top-5 stability, 6 rankers × (200 bootstrap + 25 CV folds) | `stability_secom.py` | `runs/secom_stability.json` | ~80 min |
| drift diagnostics | `drift.py` | `runs/drift.json` | ~2 min |
| rolling-origin robustness, 5 block counts | `rolling_sweep.py` | `runs/rolling_sweep.json` | 6.7 min |
| synthetic ground truth, 10 seeds | `eval_synthetic.py` | `runs/synthetic.json` | 23.0 min |

CPU only, 16 workers max, one BLAS thread per worker. `bash scripts/overnight.sh`
reproduces the lot in dependency order. Nothing was installed into the shared
environment.

## 3. KPI scorecard

Target from the catalog card: `SECOM 실데이터 AUC ≥0.75 · top-5 원인 안정성 ≥80%`.

| KPI | measured on | value | verdict |
|---|---|---|---|
| AUC ≥ 0.75 | best plain baseline `rf_all` | **0.759** [0.739, 0.779] | **met** — point estimate; the CI spans the line |
| AUC ≥ 0.75 | agent loop `agent_rf` | 0.717 [0.699, 0.735] | **not met** |
| top-5 stability ≥ 80% | agent loop, pairwise overlap, 200 bootstraps | **22.3%** | **not met** |
| top-5 stability ≥ 80% | agent loop, cluster-aware | 22.8% | **not met** |
| top-5 stability ≥ 80% | agent loop, across 25 CV training folds | 37.2% | **not met** |

The AUC target is met — **by the baseline, not by the agent loop**. The
stability target is missed by a wide margin on every protocol. Random-ranker
floor is 1.1%, so the numbers are far from chance and still far from the KPI.

## 4. What the measurements showed

### The agent loop loses to the obvious baseline on SECOM

25 folds of repeated stratified CV, byte-identical for every arm, all
preprocessing fitted inside the fold, baseline hyperparameters chosen by inner
3-fold grid search:

| arm | CV AUC | paired Δ vs `rf_all` | chrono | rolling |
|---|---|---|---|---|
| `rf_all` | 0.759 | — | 0.532 | 0.585 |
| `univar_top25_rf` | 0.730 | −0.029 [−0.041, −0.018] | 0.585 | 0.644 |
| `hgb_all` | 0.721 | −0.038 [−0.057, −0.019] | 0.482 | 0.597 |
| `agent_rf` | 0.717 | −0.042 [−0.059, −0.025] | 0.535 | 0.656 |
| `logreg_all` | 0.687 | −0.072 [−0.088, −0.056] | 0.559 | 0.513 |
| `agent_logreg` | 0.656 | −0.104 [−0.125, −0.082] | 0.511 | 0.559 |
| `majority` | 0.500 | −0.259 | 0.500 | 0.500 |

The interval excludes zero, over 25 paired folds, Wilcoxon p = 6e-5. And at
matched sparsity (25 sensors each) the loop is only **+0.010 [−0.004, +0.024]**
against picking 25 sensors one at a time — the comparison the loop most needs
to win, and it does not win it.

### Why: SECOM's signal is diffuse, so sparsity costs accuracy

The sensitivity sweep is monotone in how many sensors survive: 3.5 sensors →
0.658, 7 → 0.697, 25 → 0.740, 46 → 0.749, 82 → 0.760, all 474 → 0.766. The
loop only ties the baseline once it keeps ~45+ sensors, i.e. by declining to be
a shortlist. With the drop step disabled it converges back onto the baseline
(−0.003 [−0.017, +0.011]), which rules out a bug in the wrapper.

One design axis does pay: scoring suspects by the base model's own importance
beats held-out permutation AUC-drop at **5 of 5** matched depths (mean +0.021
AUC). With ~25 positives in an inner validation split the permutation estimate
is too noisy to rank on.

### The stability failure is mostly a sample-size wall, partly a ranker choice

| ranker | bootstrap | CV folds |
|---|---|---|
| `univariate` | 46.1% | 73.7% |
| `logreg_coef` | 42.6% | 68.8% |
| `rf_impurity` | 36.5% | 53.0% |
| `agent` (full loop) | 22.3% | 37.2% |
| `agent_no_corr` | 22.1% | 35.7% |
| `perm_only` | 20.0% | 34.1% |

Two distinct gaps, and they should not be conflated:

- **Between rankers (~24 points).** The full loop is *less* stable than a plain
  univariate ranking, because its permutation-based attribution is a noisier
  statistic at this sample size. Actionable — see §6.
- **To the KPI (~34 points, even from the best ranker).** A bootstrap resample
  omits ~37% of wafers, so each replicate sees a different ~65 fails. Which
  five of 474 weak sensors come out on top is not determined at that size.

The ablation ladder isolates each mechanism: screen+permutation 20.0%, +verify
22.1% (**+2.1**), +correlation grouping 22.3% (**+0.2**). Verification earns a
little; grouping earns almost nothing, and raw vs cluster-aware columns barely
differ — so the near-duplicate families that motivated grouping are not where
the instability lives.

### The diagnosis is measured, not asserted

`scripts/drift.py`:

- **Adversarial validation: AUC 0.9926** (shuffled-era control 0.5157). The
  same pipeline that reaches 0.759 predicting *failure* recovers *when a wafer
  was made* almost perfectly. The chronological split's halves are not two
  samples of one distribution.
- **324 of 458** surviving sensors shift significantly between the first and
  last time block (KS, Benjamini-Hochberg at FDR 0.01).
- **Fail rate drifts 14.0% → 3.5%** across five blocks, χ² p = 1e-7. The prior
  moves too, not only the features.

Forward in time every arm's rolling-origin CI includes 0.5. On a process this
non-stationary, "the top 5 causes" is not a fixed quantity estimated noisily —
it is a quantity that changes while you estimate it.

### One result points the other way, and it is the weakest here

Forward in time the ordering **inverts**: `agent_rf` is the best arm across
origins. Repeating the protocol at five block counts:

| blocks | origins | best arm | agent_rf − rf_all |
|---|---|---|---|
| 3 | 2 | `agent_rf` | +0.004 [−0.247, +0.254] |
| 4 | 3 | `hgb_all` | +0.017 [−0.027, +0.062] |
| 5 | 4 | `agent_rf` | +0.071 [−0.072, +0.214] |
| 8 | 7 | `agent_rf` | +0.011 [−0.094, +0.117] |
| 10 | 7 | `agent_rf` | +0.017 [−0.130, +0.165] |

The **sign** is stable (positive at 5 of 5; best arm at 4 of 5). The
**magnitude** is not: not one interval excludes zero, origins within a block
count share training data so even those are optimistic, and the 5-block count
used in the main table produced the *largest* of the five effects (+0.071
against a median of +0.017). Mechanism is plausible — 20 sensors give fewer
ways to lean on one that drifts than 474 do — but this is a direction, not an
effect size.

### The machinery works where its premise holds

Synthetic generator, 5 genuinely causal sensors among 200, block-correlated
noise, 10 independent datasets:

| method | top-5 recall | top-5 precision | top-5 stability |
|---|---|---|---|
| `agent` | **0.98** | 0.98 | **86.8%** |
| `rf_impurity` | 0.92 | 0.92 | 78.2% |
| `univariate` | 0.90 | 0.90 | 77.4% |

Held-out AUC: `agent_rf` 0.947, `univar_top25_rf` 0.941, `rf_all` 0.918 — the
loop is **+0.029 above** the full-sensor forest, the **opposite sign** to
SECOM, and the stability KPI is **met** (86.8%).

That contrast is the most useful thing this run produced. The loop assumes a
few sensors drive the failures. Where that premise holds it wins on accuracy
*and* stability; where the signal is spread thin over hundreds of weak,
drifting sensors, enforcing sparsity discards what the model needed.

**These synthetic numbers must never be quoted as real-data results.** SECOM
has no causal labels, so no recovery claim is made for it anywhere in the repo,
and the README's first table makes that separation explicit.

## 5. Data handling, documented and fold-internal

1,567 wafers × 590 sensors, 104 fails (6.6%, 1:14), 4.5% of cells missing,
every wafer missing at least one value, worst sensor 91.2% missing, 90 monotone
days. 116 constant sensors dropped → 474 kept. All 7 exact-duplicate groups sit
*inside* the constant set, so 0 exact duplicates remain after that drop —
near-duplicates are the real issue (179 sensors have an |r| > 0.99 partner).

Fitted inside the training fold, every time: constant/duplicate detection,
imputation medians, standardisation, missing-indicator choice, candidate screen,
permutation importance (on an inner split), bootstrap verification, baseline
hyperparameters. `tests/test_real.py::test_permuted_labels_score_at_chance`
enforces it: shuffle the labels, run the full CV, assert every arm sits near
0.5. The single documented exception is the correlation-cluster map used for the
*cluster-aware* stability variant, built from the unlabelled sensor matrix and
never used to predict.

## 6. What is left

Ordered by expected value, none of it started:

1. **Re-measure stability with `attribution="model"`.** The sweep shows it beats
   permutation on accuracy at every depth, and the stability table shows
   permutation-based rankers are the least stable. This is the one change likely
   to move the stability number materially, and it is a one-line config change
   plus ~80 min of compute. It will not reach 80%.
2. **Settle the forward-in-time reversal properly.** Needs either more wafers or
   a blocked/nested protocol with independent origins. As it stands the sign is
   consistent and the size is unestablished.
3. **A drift-aware arm.** Sample-weight by recency, or retrain per time block,
   and measure against the rolling-origin protocol rather than shuffled CV. The
   drift diagnostics make this the obvious next model, not a better ranker.
4. **Missing-indicator columns for the forest arm.** Currently only the logistic
   arm gets them; whether they help the forest is untested and listed as a limit.
5. **Interaction-only sensors.** Both screens are linear-ish or model-native, so
   a sensor mattering purely through an interaction can be dropped before
   attribution sees it. Unablated.

## 7. What needs the owner's decision

1. **The KPI card as it stands.** AUC ≥0.75 is met by a plain random forest;
   the agent loop — the thing the project is *about* — misses it, and the
   stability KPI is missed by 58 points. I have reported that rather than
   tuning toward the target. If the card should read "met", the honest routes
   are (a) restate the AUC KPI as a property of the pipeline's predictor, which
   `PredictAllReportFew` satisfies by construction at 0.759, or (b) restate the
   stability KPI against the CV-fold perturbation and a cluster-level top-5,
   where the best ranker reaches 73.7%. Both are defensible; both are a change
   of definition, not of result, and I did not make either call unilaterally.
2. **Whether the stability KPI is the right target for SECOM at all.** With 104
   fails and 70% of sensors non-stationary, ≥80% top-5 agreement may not be
   attainable by any method on this dataset. If the KPI is meant to be
   demonstrable, it needs either a different dataset or a cluster-level
   redefinition.
3. **Whether to keep `agent_logreg`.** It is the worst arm on real data
   (0.656). It is currently kept as an honest control showing the loop's cost
   depends on the base learner; it could equally be cut as noise.

## 8. Repo state

Branch `main`, pushed. `demo_smoke.py` runs on numpy alone. CI has two jobs:
the numpy-only path, and the sklearn path plus `report.py --check`. Tests: 6
smoke + 19 real-data, all green. Seven figures in `assets/`, all drawn from
`runs/`. `secom.zip` and `data/` are gitignored;
`python scripts/prepare_data.py` regenerates the extract.
