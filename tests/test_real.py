"""Tests for the real-data path (needs pandas/scikit-learn/scipy).

The centrepiece is `test_permuted_labels_score_at_chance`: shuffle the labels
so that no honest pipeline can do better than chance, then run the full
cross-validation. Anything that fitted a decision on the whole dataset --
imputation medians, constant-column detection, feature selection -- shows up
there as an AUC above 0.5. It is the one test that would have caught the class
of bug this repo was rebuilt to avoid.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from yieldrca.data import make_synthetic, secom_profile
from yieldrca.estimator import AgentRCA, PredictAllReportFew, make_logreg
from yieldrca.evaluate import Arm, chronological_split, mean_ci, paired_delta, repeated_cv
from yieldrca.preprocess import MissingIndicatorAppender, SensorCleaner, UnivariateTopK
from yieldrca.stability import _overlap_scores, bootstrap_replicates, measure, random_floor


def _small(seed=0, n=400, p=60):
    X, y, names, causal = make_synthetic(n=n, p=p, n_causal=3, fail_rate=0.12,
                                        seed=seed)
    return X, y, names, causal


# ------------------------------------------------------------ preprocessing
def test_cleaner_drops_constant_and_duplicate():
    X = np.array([[1.0, 5.0, 1.0, 2.0],
                  [2.0, 5.0, 2.0, 9.0],
                  [3.0, 5.0, 3.0, 4.0]])
    c = SensorCleaner().fit(X)
    assert c.keep_.tolist() == [0, 3]
    assert c.dropped_[1] == "constant"
    assert c.dropped_[2] == "duplicate_of_0"
    assert c.transform(X).shape == (3, 2)
    assert c.drop_counts() == {"constant": 1, "duplicate": 1}


def test_cleaner_treats_all_missing_column():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    c = SensorCleaner().fit(X)
    assert c.keep_.tolist() == [0] and c.dropped_[1] == "all_missing"


def test_cleaner_decides_from_training_rows_only():
    """A column constant in train but varying in test must still be dropped."""
    X = np.zeros((10, 3))
    X[:, 0] = np.arange(10)
    X[:, 1] = 7.0
    X[5:, 1] = np.arange(5)        # varies only in the held-out half
    X[:, 2] = np.arange(10) * 2.0
    c = SensorCleaner().fit(X[:5])
    assert 1 in c.dropped_, "constant-in-train column should be dropped"
    assert c.transform(X).shape == (10, len(c.keep_))


def test_missing_indicator_appends_expected_width():
    X = np.array([[1.0, np.nan, 3.0]] * 100)
    m = MissingIndicatorAppender(min_frac=0.01).fit(X)
    out = m.transform(X)
    assert out.shape == (100, 4) and out[:, 3].all()


def test_univariate_topk_finds_the_informative_column():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=400) < 0.3).astype(int)
    X = rng.standard_normal((400, 10))
    X[:, 4] += 3.0 * y                      # the only informative sensor
    sel = UnivariateTopK(k=2).fit(X, y)
    assert 4 in sel.keep_ and sel.transform(X).shape == (400, 2)


# ------------------------------------------------------------------ estimator
def test_agent_rca_fits_and_maps_indices_back():
    X, y, names, _ = _small()
    m = AgentRCA(base="logreg", n_screen=30, n_screen_boot=20, select_k=10,
                 stability_min=0.3, max_select=8, n_boot=4).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(y), 2)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert 1 <= len(m.selected_) <= 8
    assert m.selected_original_.max() < X.shape[1]
    # cleaned-space indices must map onto the original columns they came from
    assert np.array_equal(m.selected_original_, m.cleaner_.keep_[m.selected_])
    assert set(m.stability_) >= set(m.selected_.tolist())
    assert all(0.0 <= v <= 1.0 for v in m.stability_.values())


def test_agent_rca_survives_pure_noise():
    """No signal at all: every importance <= 0, and fit must still produce a model."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 40))
    y = (rng.uniform(size=300) < 0.1).astype(int)
    m = AgentRCA(base="logreg", n_screen=20, n_screen_boot=10, select_k=8,
                 stability_min=0.9, max_select=5, n_boot=3).fit(X, y)
    assert len(m.selected_) >= 1, "must never hand the classifier zero columns"
    p = m.predict_proba(X)
    assert p.shape == (300, 2) and np.all(np.isfinite(p))
    assert 0.0 <= float(roc_auc_score(y, p[:, 1])) <= 1.0


def test_agent_ranking_puts_survivors_first():
    """The reported ranking must lead with what survived verification."""
    X, y, names, _ = _small()
    m = AgentRCA(base="logreg", n_screen=30, select_k=10, stability_min=0.3,
                 max_select=6, n_boot=4).fit(X, y)
    r = m.ranking()
    n_sel = len(m.selected_original_)
    assert set(r[:n_sel].tolist()) == set(m.selected_original_.tolist())
    assert len(set(r.tolist())) == len(r), "ranking must not repeat a sensor"
    assert set(r.tolist()) >= set(j for j, _ in m.ranked_)
    imp = {int(j): v for j, v in m.ranked_}
    lead = [imp[int(j)] for j in r[:n_sel]]
    assert lead == sorted(lead, reverse=True), "survivors must be impact-ordered"


def test_agent_rca_report_names_selected_sensors():
    X, y, names, _ = _small()
    m = AgentRCA(base="logreg", n_screen=30, select_k=10, stability_min=0.3,
                 max_select=5, n_boot=4).fit(X, y)
    rep = m.report(names)
    assert rep.startswith("# Yield Root-Cause Report")
    assert any(names[j] in rep for j in m.selected_original_)


def test_agent_rca_recovers_planted_causes_on_synthetic():
    """Where ground truth exists, the loop must find most of it."""
    X, y, names, causal = _small(seed=3, n=800, p=80)
    m = AgentRCA(base="rf", base_kw={"n_estimators": 200}, n_screen=40,
                 n_screen_boot=30, select_k=10, stability_min=0.3,
                 max_select=10, n_boot=5).fit(X, y)
    hits = len(set(m.selected_original_.tolist()) & set(causal.tolist()))
    assert hits >= 2, f"recovered only {hits}/{len(causal)}"


def _small_rca():
    return AgentRCA(base="logreg", n_screen=30, select_k=10,
                    stability_min=0.3, max_select=6, n_boot=4)


def test_predict_all_report_few_predicts_with_every_sensor():
    """The recommended config: full-sensor prediction, loop-driven reporting."""
    X, y, names, _ = _small()
    m = PredictAllReportFew(predictor=make_logreg(),
                            rca=_small_rca()).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(axis=1), 1.0)
    # the predictor sees the cleaned matrix, not the loop's shortlist
    n_kept = len(m.cleaner_.keep_)
    assert m.predictor_[-1].n_features_in_ == n_kept
    assert len(m.selected_original_) < n_kept
    assert m.report(names).startswith("# Yield Root-Cause Report")


def test_predict_proba_is_invariant_to_the_loop():
    """The half of the docstring's claim that *is* true by construction.

    "Wrapping the loop around a predictor costs zero predictive performance"
    is a structural claim, so it is checked structurally: two instances with
    the same predictor and deliberately different loops -- different
    attribution statistic, different depth, different bootstrap count, so
    their shortlists differ -- must return bit-identical probabilities.

    The other half of that claim, that the resulting AUC equals the baseline
    row in the results, was *not* true and is not assertable: it depended on
    the default predictor matching the measured arm's construction, which it
    did not. `runs/par_few.json` measures what that was worth, and
    `test_default_predictor_matches_the_measured_baseline` pins the alignment.
    """
    X, y, _, _ = _small()
    pred = make_logreg()
    a = PredictAllReportFew(predictor=pred, rca=_small_rca()).fit(X, y)
    b = PredictAllReportFew(
        predictor=pred,
        rca=AgentRCA(base="logreg", attribution="model", n_screen=20,
                     select_k=3, stability_min=0.9, max_select=3,
                     n_boot=6)).fit(X, y)

    assert not np.array_equal(a.selected_original_, b.selected_original_), \
        "the two loops agreed, so this test would pass trivially"
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X)), \
        "the loop reached predict_proba, so the loop is not predictively free"


def test_default_predictor_matches_the_measured_baseline():
    """The default predictor must be the construction `rf_all` measures.

    The 0.759 baseline row comes from `scripts/arms.py::rf_all`, which tunes
    `min_samples_leaf` by inner CV. `PredictAllReportFew` used to default to a
    fixed `min_samples_leaf=5` forest while its docstring pointed at that row,
    so the recommended artifact was not the measured one. This asserts the two
    agree on everything that reaches the fitted model, and fails if either side
    is retuned without the other.
    """
    import arms
    from yieldrca.estimator import make_rf_tuned

    default, _ = PredictAllReportFew()._make()
    baseline = arms.rf_all()
    assert isinstance(default, type(baseline))
    assert default.param_grid == baseline.param_grid
    assert default.scoring == baseline.scoring
    assert (default.estimator["clf"].get_params()
            == baseline.estimator["clf"].get_params())
    assert (default.cv.n_splits, default.cv.shuffle, default.cv.random_state) \
        == (baseline.cv.n_splits, baseline.cv.shuffle, baseline.cv.random_state)
    # and the shared factory is what produces it, not a copy of the literals
    assert make_rf_tuned().param_grid == default.param_grid


# ------------------------------------------------------------ leakage canary
def test_permuted_labels_score_at_chance():
    """The leak detector. With shuffled labels every arm must sit near 0.5."""
    X, y, _, _ = _small(seed=1, n=400, p=60)
    rng = np.random.default_rng(0)
    y_perm = y[rng.permutation(len(y))]
    arms = [
        Arm("logreg_all",
            lambda: Pipeline([("clean", SensorCleaner()),
                              ("m", make_logreg(C=0.1))])),
        Arm("univar_top10_logreg",
            lambda: Pipeline([("clean", SensorCleaner()),
                              ("sel", UnivariateTopK(k=10)),
                              ("m", make_logreg(C=0.1))])),
    ]
    recs, _ = repeated_cv(arms, X, y_perm, n_splits=5, n_repeats=2, seed=0,
                          n_jobs=4, verbose=0)
    for name in ("logreg_all", "univar_top10_logreg"):
        aucs = [r["auc"] for r in recs if r["arm"] == name]
        m = float(np.mean(aucs))
        assert 0.38 < m < 0.62, f"{name} scores {m:.3f} on permuted labels"


def test_arms_see_identical_folds():
    X, y, _, _ = _small()
    arms = [Arm("a", lambda: Pipeline([("clean", SensorCleaner()),
                                       ("m", make_logreg())])),
            Arm("b", lambda: Pipeline([("clean", SensorCleaner()),
                                       ("m", make_logreg(C=1.0))]))]
    recs, folds = repeated_cv(arms, X, y, n_splits=5, n_repeats=1, seed=0,
                              n_jobs=4, verbose=0)
    by = {}
    for r in recs:
        by.setdefault(r["fold"], []).append((r["arm"], r["n_test"],
                                             r["n_fail_test"]))
    assert len(folds) == 5
    for f, entries in by.items():
        assert len({(n, k) for _, n, k in entries}) == 1, f"fold {f} differs"


# ------------------------------------------------------------------ statistics
def test_mean_ci_and_paired_delta():
    v = [0.70, 0.72, 0.74, 0.76, 0.78]
    s = mean_ci(v)
    assert abs(s["mean"] - 0.74) < 1e-12
    assert s["ci_lo"] < s["mean"] < s["ci_hi"] and s["n"] == 5
    d = paired_delta([0.8] * 5, v)
    assert d["mean"] > 0 and d["wins"] == 5 and d["losses"] == 0
    zero = paired_delta(v, v)
    assert zero["mean"] == 0.0 and zero["wins"] == 0


def test_chronological_split_is_ordered_and_disjoint():
    t = np.arange(100.0)[::-1]                    # deliberately unsorted
    tr, te = chronological_split(t, 0.7)
    assert len(tr) == 70 and len(te) == 30
    assert not set(tr.tolist()) & set(te.tolist())
    assert t[tr].max() <= t[te].min()


# ------------------------------------------------------------------- stability
def test_overlap_scores_endpoints():
    same = [{1, 2, 3, 4, 5}] * 4
    assert _overlap_scores(same, 5)["pairwise_overlap"] == 1.0
    assert _overlap_scores(same, 5)["consensus_freq"] == 1.0
    disjoint = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}]
    assert _overlap_scores(disjoint, 5)["pairwise_overlap"] == 0.0
    half = [{1, 2, 3, 4, 5}, {1, 2, 3, 9, 10}]
    assert abs(_overlap_scores(half, 5)["pairwise_overlap"] - 0.6) < 1e-12


def test_random_floor_and_measure_on_a_constant_ranker():
    assert abs(random_floor(474, 5) - 5 / 474) < 1e-12
    X, y, _, _ = _small()
    reps = bootstrap_replicates(y, n_boot=4, seed=0, min_pos=5)
    out = measure(lambda a, b: np.arange(a.shape[1]), X, y, reps, k=5, n_jobs=2)
    assert out["raw"]["pairwise_overlap"] == 1.0, "a fixed ranker is perfectly stable"
    assert out["n_replicates"] == 4


def test_bootstrap_replicates_respect_min_positives():
    X, y, _, _ = _small()
    for idx in bootstrap_replicates(y, n_boot=10, seed=0, min_pos=20):
        assert len(idx) == len(y) and y[idx].sum() >= 20


# ------------------------------------------------------------------- profiling
def test_secom_profile_counts_on_a_known_matrix():
    X = np.array([[1.0, 4.0, np.nan, 4.0],
                  [2.0, 4.0, 2.0, 4.0],
                  [3.0, 4.0, 3.0, 4.0]])
    y = np.array([0, 1, 0])
    p = secom_profile(X, y)
    assert p["n_wafers"] == 3 and p["n_sensors"] == 4
    assert p["n_fail"] == 1 and p["constant_sensors"] == 2
    assert p["duplicate_groups"] == 1 and p["duplicate_sensors_removable"] == 1
    assert p["sensors_with_any_missing"] == 1
    assert p["rows_with_any_missing"] == 1


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    for n, f in fns:
        f()
        print(f"ok {n}")
    print(f"ok  ({len(fns)} tests)")


def test_weekend_audit_catches_an_invented_number():
    """The coverage scan must fail on a number no run produced.

    `audit_weekend.py` used to check only that registered claims appear, which
    cannot catch a number nobody registered -- and twice this weekend a typed
    number passed it. The scan now runs the other direction, so this plants an
    invented figure and asserts it is flagged, then checks a structural value
    and a real measurement are not.
    """
    import audit_weekend as aw

    rows = [("a real claim", "some_run", "0.729 [0.710, 0.748]"),
            ("another", "some_run", "94.2%")]

    bad = dict(aw.coverage("The loop scored 0.729 [0.710, 0.748] and "
                           "0.888 on the second axis.", rows))
    assert "0.888" in bad, "an invented number was not flagged"
    assert "0.729" not in bad, "a registered number was wrongly flagged"

    # Code blocks carry commands, not results, and must be skipped.
    fenced = "```\nscripts/foo.py --n-boot 37\n```\n"
    assert not aw.coverage(fenced, rows), "a flag inside a code block was flagged"

    # "top-5" must not read as the number -5, and ISO dates must not tokenise.
    prose = "The top-5 on 2026-09-05 for sensor_059 reached 94.2%."
    assert not aw.coverage(prose, rows), (
        f"tokeniser artefacts leaked through: {aw.coverage(prose, rows)}")
