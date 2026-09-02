"""Role-specialized agents for yield root-cause analysis.

The pattern mirrors a Planner / worker / Critic multi-agent system: each
agent owns one tool and one question. The Orchestrator (pipeline.py) routes
between them. The LLM Reporter is optional — with no API key it emits a
deterministic templated report, so the pipeline never hard-depends on a model.
"""
from __future__ import annotations
import numpy as np
from .model import permutation_importance, roc_auc


class SensorAgent:
    """Owns the classifier. Answers: which sensors move the fail probability?"""

    def __init__(self, model):
        self.model = model

    def attribute(self, X, y, top_k=8):
        imp = permutation_importance(self.model, X, y, roc_auc)
        order = np.argsort(imp)[::-1][:top_k]
        return [(int(i), float(imp[i])) for i in order if imp[i] > 0]


class CorrelatorAgent:
    """Owns correlation tooling. Answers: are suspects distinct or redundant?"""

    def cluster(self, X, idx, thresh=0.7):
        cols = np.where(np.isnan(X), np.nanmean(X, axis=0), X)[:, idx]
        c = np.corrcoef(cols.T)
        clusters, seen = [], set()
        for a in range(len(idx)):
            if a in seen:
                continue
            grp = [idx[a]]
            seen.add(a)
            for b in range(a + 1, len(idx)):
                if b not in seen and abs(c[a, b]) >= thresh:
                    grp.append(idx[b])
                    seen.add(b)
            clusters.append(grp)
        return clusters


class VerifierAgent:
    """Owns validation. Answers: is a suspect stable, or a fluke of one split?"""

    def stability(self, model_ctor, X, y, idx, n_boot=8, top_k=8, seed=0):
        rng = np.random.default_rng(seed)
        hits = {i: 0 for i in idx}
        for _ in range(n_boot):
            b = rng.integers(0, len(y), len(y))
            m = model_ctor().fit(X[b], y[b])
            imp = permutation_importance(m, X[b], y[b], roc_auc, n_repeats=2)
            top = set(np.argsort(imp)[::-1][:top_k].tolist())
            for i in idx:
                hits[i] += int(i in top)
        return {i: hits[i] / n_boot for i in idx}


class ReporterAgent:
    """Optional LLM writer; deterministic template fallback with no key."""

    def write(self, ranked, clusters, stability, names):
        lines = ["# Yield Root-Cause Report", ""]
        lines.append(f"Identified {len(ranked)} suspect sensors "
                     f"across {len(clusters)} independent group(s).\n")
        for rank, (i, score) in enumerate(ranked, 1):
            conf = stability.get(i, 0.0)
            lines.append(
                f"{rank}. **{names[i]}** — impact {score:.3f}, "
                f"stability {conf:.0%}"
            )
        lines.append("\n## Independent root-cause groups")
        for g in clusters:
            lines.append("- " + ", ".join(names[i] for i in g))
        lines.append("\n_Reporter ran in offline/template mode "
                     "(set OPENAI_API_KEY for an LLM narrative)._")
        return "\n".join(lines)
