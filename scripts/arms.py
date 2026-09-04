"""The arms every SECOM experiment shares, in one place.

Baselines get their hyperparameters chosen by an **inner** 3-fold CV on each
outer training fold, so no baseline number in this repo is the best of a grid
scored on the test folds. The agent loop's structural settings cannot be tuned
that way (they change what the pipeline *reports*, not just how it scores), so
they are pre-registered at one operating point -- chosen for deliverable size,
"a report of at most 25 suspects with at least 30% bootstrap support", not for
AUC -- and the full sensitivity sweep is published in
``runs/secom_loop_sweep.json`` instead of hidden.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yieldrca.estimator import AgentRCA
from yieldrca.evaluate import Arm
from yieldrca.preprocess import (
    MissingIndicatorAppender,
    SensorCleaner,
    UnivariateTopK,
)

SEED = 0
INNER = 3
AGENT_BASE_KW = {"n_estimators": 300}

# Pre-registered agent-loop operating point (see module docstring).
AGENT_CFG = dict(
    screen="model", attribution="permutation", n_screen=150, n_screen_boot=60,
    select_k=40, top_k=5, stability_min=0.3, max_select=25, n_boot=12,
    corr_thresh=0.9, n_inner=3, n_repeats=3, random_state=SEED,
)


def _inner():
    return StratifiedKFold(n_splits=INNER, shuffle=True, random_state=SEED)


def _tuned(pipe, grid):
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=_inner(), n_jobs=1,
                        refit=True)


def majority():
    return Pipeline([("clf", DummyClassifier(strategy="prior"))])


def logreg_all():
    pipe = Pipeline([
        ("clean", SensorCleaner()),
        ("missind", MissingIndicatorAppender(min_frac=0.01)),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=3000,
                                   random_state=SEED)),
    ])
    return _tuned(pipe, {"clf__C": [0.003, 0.03, 0.3]})


def hgb_all():
    from sklearn.ensemble import HistGradientBoostingClassifier

    pipe = Pipeline([
        ("clean", SensorCleaner()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200, min_samples_leaf=20, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=SEED)),
    ])
    return _tuned(pipe, {"clf__learning_rate": [0.03, 0.05, 0.1],
                         "clf__max_leaf_nodes": [7, 15, 31]})


def rf_all():
    pipe = Pipeline([
        ("clean", SensorCleaner()),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=1, random_state=SEED)),
    ])
    return _tuned(pipe, {"clf__min_samples_leaf": [1, 5, 10]})


def univar_rf(k=25):
    """Naive selection at the agent loop's budget, fitted inside the fold."""
    return Pipeline([
        ("clean", SensorCleaner()),
        ("select", UnivariateTopK(k=k)),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=1, random_state=SEED)),
    ])


def agent(base="rf", **over):
    cfg = dict(AGENT_CFG)
    cfg.update(over)
    kw = AGENT_BASE_KW if base == "rf" else {}
    return AgentRCA(base=base, base_kw=kw, **cfg)


def _agent_extras(est):
    return {
        "n_selected": int(len(est.selected_)),
        "n_candidates": int(est.n_candidates_),
        "selected": est.selected_original_.tolist(),
        "top5": [j for j, _ in est.ranked_[:5]],
        "n_groups": int(len(est.groups_)),
    }


def _tuned_extras(est):
    return {"chosen": {k: v for k, v in getattr(est, "best_params_", {}).items()}}


def main_arms():
    """The headline comparison: 4 baselines, 1 selection control, 2 agent arms."""
    return [
        Arm("majority", majority, "majority class (DummyClassifier)", "baseline"),
        Arm("logreg_all", logreg_all,
            "logistic regression, all sensors (C tuned inner-CV)", "baseline",
            extras=_tuned_extras),
        Arm("hgb_all", hgb_all,
            "hist gradient boosting, all sensors (tuned inner-CV)", "baseline",
            extras=_tuned_extras),
        Arm("rf_all", rf_all,
            "random forest, all sensors (tuned inner-CV)", "baseline",
            extras=_tuned_extras),
        Arm("univar_top25_rf", univar_rf,
            "univariate top-25 sensors -> random forest", "control"),
        Arm("agent_rf", lambda: agent("rf"),
            "agent loop (RF probe + RF on survivors)", "agent",
            extras=_agent_extras),
        Arm("agent_logreg", lambda: agent("logreg"),
            "agent loop (logistic probe + logistic on survivors)", "agent",
            extras=_agent_extras),
    ]
