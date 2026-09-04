"""``AgentRCA``: the plan -> execute -> verify agent loop as an sklearn estimator.

Wrapping the loop in ``fit``/``predict_proba`` is what makes the headline claim
testable. Everything the loop does -- cleaning, screening, attribution,
correlation grouping, bootstrap verification, final refit -- happens inside
``fit``, so dropping it into ``cross_validate`` against a plain classifier
measures exactly one thing: *does routing agents over the sensors beat handing
all of them to the model?* If the loop peeked at the test fold anywhere, that
comparison would be worthless, so it cannot.

    SensorAgent      screen + held-out permutation importance -> ranked suspects
    CorrelatorAgent  group near-identical sensors -> one representative each
    VerifierAgent    re-derive the ranking on bootstrap resamples -> stability
    (drop)           suspects below `stability_min` are discarded
    ReporterAgent    render survivors + groups (yieldrca.agents)
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .attribution import (
    _pos_score,
    correlation_clusters,
    permutation_importance_heldout,
    screen_model,
    screen_multivariate,
    screen_univariate,
)
from .preprocess import MissingIndicatorAppender, SensorCleaner


def make_logreg(C=0.03, seed=0):
    """Median-impute -> standardise -> missing-indicator -> L2 logistic."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                   random_state=seed)),
    ])


def make_logreg_missind(C=0.03, seed=0):
    return Pipeline([
        ("missind", MissingIndicatorAppender(min_frac=0.01)),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000,
                                   random_state=seed)),
    ])


def make_hgb(learning_rate=0.05, max_leaf_nodes=15, max_iter=200, seed=0):
    """Histogram gradient boosting -- consumes NaN natively, so no imputer."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter, min_samples_leaf=20, l2_regularization=1.0,
        class_weight="balanced", early_stopping=False, random_state=seed,
    )


def make_rf(n_estimators=500, min_samples_leaf=5, max_features="sqrt", seed=0):
    """Median-impute -> random forest. The strongest plain baseline on SECOM."""
    from sklearn.ensemble import RandomForestClassifier

    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features=max_features, class_weight="balanced_subsample",
            n_jobs=1, random_state=seed)),
    ])


BASE_FACTORIES = {
    "logreg": make_logreg,
    "logreg_missind": make_logreg_missind,
    "hgb": make_hgb,
    "rf": make_rf,
}


class AgentRCA(BaseEstimator, ClassifierMixin):
    """Agent loop over sensors, then a classifier on the survivors.

    Parameters
    ----------
    base : {"logreg", "logreg_missind", "hgb", "rf"}
        Final classifier, also used as the attribution probe.
    attribution : {"permutation", "model"}
        How suspects are scored. ``"permutation"`` is the held-out AUC drop --
        principled, but with ~25 positives in an inner validation split it is
        also noisy. ``"model"`` averages the base model's own importance over
        the inner splits instead: cheaper, lower variance, and it cannot be
        read as a causal effect.
    screen : {"logreg", "model", "univariate"}
        Candidate-pool screen. ``"logreg"`` ranks by |standardised logistic
        coefficient|; ``"model"`` uses the base model's own importance, which
        matters when the base is a forest and the screen is not; ``"univariate"``
        is the naive per-sensor control.
    n_screen, n_screen_boot : int
        Candidate-pool size for the full ranking and for each bootstrap replay.
    select_k : int
        Rank depth that counts as "selected" in each bootstrap replay. This is
        the *predictive* vote depth and is deliberately separate from
        ``top_k``, which is only the reporting depth: a sensor can be a useful
        predictor without ever being the single most important one.
    top_k : int
        Reporting depth. ``stability_top_k_`` records how often each survivor
        reached this depth -- the quantity the top-5 stability KPI is about.
    stability_min : float
        Bootstrap selection frequency a suspect must reach to survive the drop
        step (classic stability selection with threshold pi).
    max_select : int
        Cap on selected sensors handed to the final classifier.
    n_boot : int
        Bootstrap resamples the VerifierAgent runs.
    report_tau : float or None
        Null-calibrated bar a suspect must clear to be *reported*, as opposed to
        ``stability_min``, which only decides what the final classifier is
        handed. ``None`` reproduces the historical behaviour: everything in
        ``selected_`` is reported and the report is never empty.

        Set it and ``reported_`` may come back empty, which is the whole point.
        The intended value is ``tau(alpha)`` from `scripts/abstain.py`: the
        ``(1 - alpha)`` quantile of the largest bootstrap support this same loop
        produces on permuted labels. Because that null is measured rather than
        assumed, "no sensor here is above noise" becomes an outcome the pipeline
        can actually reach -- `runs/null_fdr.json` shows it otherwise cannot,
        at any setting of ``stability_min``.

        Prediction is deliberately untouched: ``selected_`` and
        ``predict_proba`` behave identically either way, so turning abstention
        on cannot move an AUC and the two claims stay separable.
    corr_thresh : float
        |r| at or above which two sensors are treated as one signal.
    """

    def __init__(self, base="hgb", base_kw=None, screen="logreg",
                 attribution="permutation", n_screen=60, n_screen_boot=40,
                 select_k=20, top_k=5, stability_min=0.5, max_select=25,
                 n_boot=12, corr_thresh=0.9, n_inner=3, n_repeats=3,
                 report_tau=None, random_state=0):
        self.base = base
        self.base_kw = base_kw
        self.screen = screen
        self.attribution = attribution
        self.n_screen = n_screen
        self.n_screen_boot = n_screen_boot
        self.select_k = select_k
        self.top_k = top_k
        self.stability_min = stability_min
        self.max_select = max_select
        self.n_boot = n_boot
        self.corr_thresh = corr_thresh
        self.n_inner = n_inner
        self.n_repeats = n_repeats
        self.report_tau = report_tau
        self.random_state = random_state

    # -- helpers ---------------------------------------------------------
    def _make_base(self, seed=None):
        kw = dict(self.base_kw or {})
        kw.setdefault("seed", self.random_state if seed is None else seed)
        return BASE_FACTORIES[self.base](**kw)

    def _fit_base(self, X, y, seed=None):
        return clone(self._make_base(seed)).fit(X, y)

    def _screen(self, Xc, y, n_keep, seed):
        if self.screen == "model":
            return screen_model(lambda a, b: self._fit_base(a, b, seed=seed),
                                Xc, y, n_keep=n_keep)
        if self.screen == "univariate":
            return screen_univariate(Xc, y, n_keep=n_keep)
        return screen_multivariate(Xc, y, n_keep=n_keep, seed=seed)

    def _rank(self, Xc, y, n_screen, n_inner, n_repeats, seed):
        """One pass of SensorAgent: screen -> importance over the candidate pool."""
        cand, w = self._screen(Xc, y, n_screen, seed)
        if self.attribution == "model":
            imp = np.zeros(Xc.shape[1])
            rng = np.random.default_rng(seed)
            for r in range(max(1, n_inner)):
                sub = rng.choice(len(y), len(y), replace=True) if r else np.arange(len(y))
                if y[sub].sum() < 5:
                    continue
                _, wr = screen_model(lambda a, b: self._fit_base(a, b, seed=seed + r),
                                     Xc[sub], y[sub], n_keep=1)
                imp[cand] += wr[cand]
            return imp / max(1, n_inner)
        return permutation_importance_heldout(
            lambda a, b: self._fit_base(a, b, seed=seed), Xc, y, cand,
            n_splits=n_inner, n_repeats=n_repeats, seed=seed)

    # -- sklearn API -----------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        rs = self.random_state

        # --- 0. clean (fold-internal) ----------------------------------
        self.cleaner_ = SensorCleaner().fit(X)
        Xc = self.cleaner_.transform(X)
        keep = self.cleaner_.keep_

        # --- 1. SensorAgent --------------------------------------------
        imp = self._rank(Xc, y, self.n_screen, self.n_inner, self.n_repeats, rs)
        order = np.argsort(imp)[::-1]
        suspects = [int(j) for j in order if imp[j] > 0][: self.max_select * 3]
        if not suspects:
            # On data with no signal, permuting any column can *improve* the
            # held-out AUC, so every importance comes back <= 0 and there is
            # nothing to select. Take the least-bad candidates rather than
            # handing the classifier an empty matrix.
            suspects = [int(j) for j in order[: max(self.top_k, 1)]]

        # --- 2. CorrelatorAgent: one representative per signal family ---
        groups = correlation_clusters(Xc, suspects, thresh=self.corr_thresh)
        reps = [max(g, key=lambda j: imp[j]) for g in groups]
        reps.sort(key=lambda j: -imp[j])

        # --- 3. VerifierAgent: replay the ranking on bootstrap resamples -
        rng = np.random.default_rng(rs)
        hits = {int(j): 0 for j in reps}
        hits_top = {int(j): 0 for j in reps}
        n_eff = 0
        for b in range(self.n_boot):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].sum() < 5:
                continue
            imp_b = self._rank(Xc[idx], y[idx], self.n_screen_boot,
                               max(1, self.n_inner - 2), max(1, self.n_repeats - 1),
                               rs + 100 + b)
            order_b = np.argsort(imp_b)[::-1]
            sel_b = set(int(j) for j in order_b[: self.select_k] if imp_b[j] > 0)
            top_b = set(int(j) for j in order_b[: self.top_k] if imp_b[j] > 0)
            for j in hits:
                hits[j] += int(j in sel_b)
                hits_top[j] += int(j in top_b)
            n_eff += 1
        self.stability_ = {j: (h / n_eff if n_eff else 0.0) for j, h in hits.items()}
        self.stability_top_k_ = {j: (h / n_eff if n_eff else 0.0)
                                 for j, h in hits_top.items()}

        # --- 4. drop the unstable, then refit --------------------------
        surv = [j for j in reps if self.stability_[j] >= self.stability_min]
        if not surv:  # never return empty-handed; fall back to the top reps
            surv = reps[: min(5, len(reps))]
        surv = surv[: self.max_select]

        self.selected_ = np.asarray(sorted(surv), dtype=int)          # cleaned space
        self.selected_original_ = keep[self.selected_]                 # original space

        # What gets *reported* is a separate decision from what gets predicted
        # with, and unlike `selected_` it is allowed to be empty. Ordered by
        # impact, because a report is read top-down.
        if self.report_tau is None:
            rep_j = [int(j) for j in self.selected_]
        else:
            rep_j = [int(j) for j in reps
                     if self.stability_[int(j)] >= self.report_tau]
        rep_j.sort(key=lambda j: -imp[j])
        self.reported_ = np.asarray(rep_j, dtype=int)
        self.reported_original_ = keep[self.reported_] if len(rep_j) else \
            np.asarray([], dtype=int)
        self.abstained_ = bool(len(rep_j) == 0)
        self.importance_ = {int(j): float(imp[j]) for j in reps}
        self.groups_ = [[int(keep[j]) for j in g] for g in groups]
        self.ranked_ = [(int(keep[j]), float(imp[j])) for j in reps]
        self.n_candidates_ = len(reps)

        self.model_ = self._fit_base(Xc[:, self.selected_], y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = self.cleaner_.transform(X)[:, self.selected_]
        p1 = _pos_score(self.model_, Xc)
        return np.column_stack([1 - p1, p1])

    def decision_function(self, X):
        return self.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # -- reporting -------------------------------------------------------
    def ranking(self):
        """The sensor ranking this loop reports, as original column indices.

        Survivors of the verify-and-drop step first, ordered by impact, then
        the dropped candidates so a top-k stays well defined when few survive.
        This -- not ``ranked_``, which is written before anything is dropped --
        is what a stability measurement should consume, or the VerifierAgent
        drops out of the comparison without anyone noticing.
        """
        imp = {int(j): v for j, v in self.ranked_}
        surv = sorted((int(j) for j in self.selected_original_),
                      key=lambda j: -imp.get(j, 0.0))
        seen = set(surv)
        rest = [int(j) for j, _ in self.ranked_ if int(j) not in seen]
        return np.asarray(surv + rest, dtype=int)

    def report(self, names=None):
        from .agents import ReporterAgent

        names = names or [f"sensor_{i:03d}" for i in range(self.cleaner_.n_features_in_)]
        keep = self.cleaner_.keep_
        ranked = [(int(keep[j]), float(self.importance_[j]))
                  for j in self.reported_ if j in self.importance_]
        ranked.sort(key=lambda t: -t[1])
        stab = {int(keep[j]): v for j, v in self.stability_.items()}
        sel = set(int(j) for j in self.reported_original_)
        groups = [g for g in self.groups_ if sel & set(g)]
        return ReporterAgent().write(ranked, groups, stab, names,
                                     tau=self.report_tau)


class PredictAllReportFew(BaseEstimator, ClassifierMixin):
    """Predict with every sensor; report with the agent loop.

    This is the configuration the SECOM measurements actually point at, made
    executable so it does not stay advice. Selecting sensors costs held-out
    AUC there -- monotonically, over the whole sweep -- so the classifier keeps
    all of them, while the loop runs alongside for its ranked,
    stability-scored suspect list.

    Its held-out AUC is the full-sensor model's *by construction*: the loop
    never touches ``predict_proba``. There is nothing to measure about that
    number beyond the baseline row already in the results, which is the point
    -- used this way the loop costs nothing predictive, and what it produces is
    a work order rather than a claim.
    """

    def __init__(self, predictor=None, rca=None):
        self.predictor = predictor
        self.rca = rca

    def _make(self):
        pred = self.predictor if self.predictor is not None else Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clf", make_rf()[-1]),
        ])
        return pred, (self.rca if self.rca is not None else AgentRCA(base="rf"))

    def fit(self, X, y):
        from .preprocess import SensorCleaner

        X = np.asarray(X, dtype=np.float64)
        pred, rca = self._make()
        self.cleaner_ = SensorCleaner().fit(X)
        self.predictor_ = clone(pred).fit(self.cleaner_.transform(X), y)
        self.rca_ = clone(rca).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Xc = self.cleaner_.transform(np.asarray(X, dtype=np.float64))
        p1 = _pos_score(self.predictor_, Xc)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def report(self, names=None):
        return self.rca_.report(names)

    @property
    def selected_original_(self):
        return self.rca_.selected_original_

    @property
    def reported_original_(self):
        return self.rca_.reported_original_

    @property
    def abstained_(self):
        return self.rca_.abstained_
