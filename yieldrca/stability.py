"""Top-5 stability: the definition, then the measurement.

**The definition, fixed before any number was computed.**

Let a *resampling scheme* produce B replicates of the wafer set. For replicate
``b`` the whole ranking pipeline is re-run using only that replicate's rows --
cleaning, imputation, screening, attribution, verification, all of it -- and
emits an ordered sensor ranking whose first five entries are ``T_b``.

* **Primary -- mean pairwise top-5 overlap**

      S_pair = mean over all B(B-1)/2 unordered pairs (b, b') of
               |T_b intersect T_b'| / 5

  This is the number the KPI is read against. It has no reference set, so it
  cannot be inflated by choosing the reference after seeing the resamples, and
  it is exactly "how often would two engineers running this on two samples of
  the same fab get the same five sensors".

* **Secondary -- consensus top-5 selection frequency**

      C = the five sensors appearing in the most T_b
      S_cons = (1/5) * sum over j in C of |{b : j in T_b}| / B

  Reported second because it is the more flattering of the two: C is chosen
  after seeing every replicate, so S_cons >= S_pair essentially always.

* **Cluster-aware variants** of both. SECOM contains 179 sensors that have a
  partner correlated above |r| = 0.99. Two near-identical signals swap places
  between resamples, which costs the raw score without meaning anything
  physically -- and an engineer handed either one goes and looks at the same
  piece of equipment. So each metric is also computed after mapping every
  sensor to its correlation cluster (single-link, |r| >= ``cluster_thresh``),
  with ``T_b`` becoming a set of cluster ids. The cluster map is built once
  from the **unlabelled** sensor matrix, so no label information crosses
  replicates; it is used for interpretation only and never for prediction.

A random ranker over ``p`` effective sensors scores ``S_pair ~ 5/p``, which for
SECOM's 474 surviving columns is about 1%. That is the floor these numbers sit
against, and it is reported alongside them.
"""
from __future__ import annotations

import itertools

import numpy as np
from joblib import Parallel, delayed

from .attribution import correlation_clusters
from .preprocess import SensorCleaner


def cluster_map(X, thresh=0.99):
    """Map every original sensor index -> cluster id (label-free)."""
    cleaner = SensorCleaner().fit(X)
    Xc = cleaner.transform(X)
    groups = correlation_clusters(Xc, np.arange(Xc.shape[1]), thresh=thresh)
    cmap = {}
    for cid, g in enumerate(groups):
        for j in g:
            cmap[int(cleaner.keep_[j])] = cid
    nxt = len(groups)
    for j in range(X.shape[1]):
        if j not in cmap:
            cmap[j] = nxt
            nxt += 1
    return cmap, len(groups)


def _overlap_scores(top_sets, k):
    pairs = [len(a & b) / k for a, b in itertools.combinations(top_sets, 2)]
    freq: dict[int, int] = {}
    for s in top_sets:
        for j in s:
            freq[j] = freq.get(j, 0) + 1
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:k]
    cons = sum(c for _, c in top) / (k * len(top_sets)) if top_sets else 0.0
    return {
        "pairwise_overlap": float(np.mean(pairs)) if pairs else float("nan"),
        "pairwise_overlap_sd": float(np.std(pairs, ddof=1)) if len(pairs) > 1 else 0.0,
        "consensus_freq": float(cons),
        "consensus_members": [int(j) for j, _ in top],
        "consensus_member_freq": [c / len(top_sets) for _, c in top],
        "n_distinct_sensors": len(freq),
    }


def bootstrap_replicates(y, n_boot=200, seed=0, min_pos=10):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() >= min_pos:
            out.append(idx)
    return out


def cv_train_replicates(X, y, n_splits=5, n_repeats=5, seed=0):
    from sklearn.model_selection import RepeatedStratifiedKFold

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    return [tr for tr, _ in cv.split(X, y)]


def _rank_one(ranker, X, y, idx, k):
    r = ranker(X[idx], y[idx])
    return [int(j) for j in r[:k]]


def measure(ranker, X, y, replicates, k=5, cmap=None, n_jobs=16, verbose=0):
    """Run ``ranker(X_b, y_b) -> ordered sensor indices`` on every replicate."""
    tops = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_rank_one)(ranker, X, y, idx, k) for idx in replicates)
    tops = [t for t in tops if len(t) == k]
    raw = _overlap_scores([set(t) for t in tops], k)
    out = {"k": k, "n_replicates": len(tops), "raw": raw,
           "top_sets": [sorted(t) for t in tops]}
    if cmap is not None:
        cl = [set(cmap[j] for j in t) for t in tops]
        out["cluster"] = _overlap_scores(cl, k)
    return out


def random_floor(p_eff, k=5):
    """Expected pairwise overlap for a uniformly random top-k over p_eff sensors."""
    return k / p_eff
