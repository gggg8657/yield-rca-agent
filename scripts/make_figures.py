"""Render the README figures from runs/*.json.

    python scripts/make_figures.py

Four figures, each one answering a question the tables answer more precisely:

* ``fig_secom_auc.png``  -- who beats whom, and by more than the CI?
* ``fig_sparsity.png``   -- what does selecting fewer sensors cost?
* ``fig_stability.png``  -- how far is top-5 stability from the KPI?
* ``fig_protocol.png``   -- how much of the AUC survives a chronological split?

Every value plotted is read from the run JSONs, so the figures cannot drift
from the tables. Categorical colours are the first three slots of the
validated default palette (blue / orange / aqua), which clear the all-pairs
CVD and normal-vision floors; aqua sits below 3:1 against the surface, so every
mark also carries a direct value label rather than relying on colour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8981"
GRID = "#e6e5e1"
SERIES = {"baseline": "#2a78d6", "control": "#eb6834", "agent": "#1baf7a"}
KPI_AUC, KPI_STAB = 0.75, 0.80

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "figure.dpi": 160,
})


def read(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _legend_below(fig, handles, ncol=3):
    """Legends go under the figure. Inside the plot they land on the data."""
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handletextpad=0.4,
               columnspacing=1.8, borderpad=0,
               bbox_to_anchor=(0.5, -0.02))


def _ref_lane(ax, n_rows, pad=1.25):
    """Reserve a strip above the top row for reference-line labels.

    Reference labels put *at* the top row collide with the data; put above the
    axes they collide with the title. A dedicated lane avoids both.
    """
    ax.set_ylim(-0.65, n_rows - 1 + pad)
    return n_rows - 1 + 0.62


def fig_auc(ev, out):
    auc = ev["auc"]["per_arm"]
    kinds = {k: v["kind"] for k, v in ev["arms"].items()}
    names = sorted(auc, key=lambda a: auc[a]["mean"])
    y = range(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _grid(ax)
    for i, n in enumerate(names):
        v = auc[n]
        c = SERIES[kinds[n]]
        ax.barh(i, v["mean"], height=0.58, color=c, zorder=3,
                edgecolor=SURFACE, linewidth=1.2)
        # 2px surface halo under the interval so it stays legible where it
        # overlaps the fill, then the interval itself with end caps
        ax.plot([v["ci_lo"], v["ci_hi"]], [i, i], color=SURFACE, linewidth=4.2,
                solid_capstyle="butt", zorder=4)
        ax.errorbar(v["mean"], i,
                    xerr=[[v["mean"] - v["ci_lo"]], [v["ci_hi"] - v["mean"]]],
                    fmt="none", ecolor=INK_2, elinewidth=1.5, capsize=3,
                    capthick=1.5, zorder=5)
        ax.text(v["ci_hi"] + 0.006, i, f"{v['mean']:.3f}", va="center",
                ha="left", fontsize=8.5, color=INK)
    lane = _ref_lane(ax, len(names))
    ax.axvline(KPI_AUC, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=5)
    ax.text(KPI_AUC + 0.004, lane, f"KPI {KPI_AUC:.2f}", fontsize=8,
            color=INK_2, va="center", ha="left")
    ax.set_yticks(list(y), names, fontsize=8.5)
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel(f"ROC-AUC, {ev['auc']['n_folds']} folds (bar = mean, "
                  "line = 95% CI)")
    ax.set_title("SECOM: the agent loop against the obvious baselines",
                 fontsize=10.5, loc="left", pad=10)
    _legend_below(fig, [Line2D([], [], marker="s", linestyle="", markersize=7,
                               color=SERIES[k], label=k)
                        for k in ("baseline", "control", "agent")], ncol=3)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_sparsity(sw, out):
    auc = sw["auc"]["per_arm"]
    base = auc["rf_all"]
    pts = {}
    for name, meta in sw["arms"].items():
        if name not in sw["n_selected_mean"] or not name.startswith("agent_"):
            continue
        mode = meta["meta"].get("attribution", "?")
        pts.setdefault(mode, []).append(
            (sw["n_selected_mean"][name], auc[name]["mean"],
             auc[name]["ci_lo"], auc[name]["ci_hi"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _grid(ax, axis="both")
    ax.axhspan(base["ci_lo"], base["ci_hi"], color=GRID, zorder=1)
    ax.axhline(base["mean"], color=INK_MUTED, linewidth=1.3, zorder=2)
    ax.text(1.15, base["mean"], f" rf_all, all sensors  {base['mean']:.3f}",
            fontsize=8.5, color=INK_2, va="bottom", ha="left")
    ax.axhline(KPI_AUC, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=2)
    ax.text(1.15, KPI_AUC, f" KPI {KPI_AUC:.2f}", fontsize=8, color=INK_2,
            va="top", ha="left")
    colors = ["#2a78d6", "#eb6834"]
    handles = []
    for (mode, vals), c in zip(sorted(pts.items()), colors):
        vals.sort()
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        for x, y, lo, hi in vals:
            ax.plot([x, x], [lo, hi], color=c, linewidth=1.4, alpha=0.55,
                    zorder=3)
        ax.plot(xs, ys, color=c, linewidth=2, zorder=4,
                label=f"attribution = {mode}")
        handles.append(Line2D([], [], marker="o", linestyle="-", linewidth=2,
                              markersize=7, color=c,
                              label=f"attribution = {mode}"))
        ax.scatter(xs, ys, s=42, color=c, zorder=5, edgecolor=SURFACE,
                   linewidth=1.6)
        for x, y, *_ in vals:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, -13), ha="center", fontsize=7.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("sensors handed to the final classifier (mean over folds)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Every sensor the loop drops costs accuracy",
                 fontsize=10.5, loc="left", pad=10)
    _legend_below(fig, handles, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_stability(st, out):
    order = sorted(st["rankers"],
                   key=lambda r: st["rankers"][r]["bootstrap"]["raw"]["pairwise_overlap"])
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(order) + 1.9))
    _grid(ax)
    h = 0.36
    for i, name in enumerate(order):
        r = st["rankers"][name]
        for k, (scheme, c) in enumerate((("bootstrap", "#2a78d6"),
                                         ("cv_train", "#eb6834"))):
            v = r[scheme]["raw"]["pairwise_overlap"]
            ax.barh(i + (0.5 - k) * h, v, height=h - 0.04, color=c, zorder=3,
                    edgecolor=SURFACE, linewidth=1.2)
            ax.text(v + 0.008, i + (0.5 - k) * h, f"{100*v:.0f}%", va="center",
                    ha="left", fontsize=8, color=INK)
    lane = _ref_lane(ax, len(order))
    ax.axvline(KPI_STAB, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=5)
    ax.text(KPI_STAB + 0.008, lane, f"KPI {KPI_STAB:.0%}", fontsize=8,
            color=INK_2, va="center", ha="left")
    ax.axvline(st["random_floor_raw"], color=INK_MUTED, linewidth=1.0,
               zorder=5)
    ax.text(st["random_floor_raw"] + 0.008, lane,
            f"random ranker {st['random_floor_raw']:.1%}", fontsize=7.5,
            color=INK_MUTED, va="center", ha="left")
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("mean pairwise top-5 overlap")
    ax.set_title("Top-5 stability, and how much the perturbation matters",
                 fontsize=10.5, loc="left", pad=10)
    _legend_below(fig, [
        Line2D([], [], marker="s", linestyle="", markersize=7, color="#2a78d6",
               label=f"bootstrap resamples (B="
                     f"{st['schemes']['bootstrap']['n_replicates']}), ~40% shared rows"),
        Line2D([], [], marker="s", linestyle="", markersize=7, color="#eb6834",
               label=f"CV training folds (n="
                     f"{st['schemes']['cv_train']['n_replicates']}), 75% shared rows"),
    ], ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_protocol(ev, out):
    auc = ev["auc"]["per_arm"]
    ch = ev["chronological"]
    names = [a for a in sorted(auc, key=lambda a: auc[a]["mean"])
             if a != "majority"]
    fig, ax = plt.subplots(figsize=(6.8, 0.55 * len(names) + 1.9))
    _grid(ax)
    for i, n in enumerate(names):
        a, b = auc[n]["mean"], ch[n]["auc"]
        ax.plot([b, a], [i, i], color=GRID, linewidth=3, solid_capstyle="round",
                zorder=2)
        ax.scatter([a], [i], s=52, color="#2a78d6", zorder=4,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.scatter([b], [i], s=52, color="#eb6834", zorder=4,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.text(a + 0.008, i, f"{a:.3f}", va="center", ha="left", fontsize=8,
                color=INK)
        ax.text(b - 0.008, i, f"{b:.3f}", va="center", ha="right", fontsize=8,
                color=INK)
    lane = _ref_lane(ax, len(names))
    ax.axvline(0.5, color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.text(0.5 - 0.004, lane, "chance", fontsize=7.5, color=INK_MUTED,
            va="center", ha="right")
    ax.axvline(KPI_AUC, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=1)
    ax.text(KPI_AUC + 0.004, lane, f"KPI {KPI_AUC:.2f}", fontsize=8,
            color=INK_2, va="center", ha="left")
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.set_xlim(0.40, 0.86)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Shuffled CV flatters SECOM; forward in time, little survives",
                 fontsize=10.5, loc="left", pad=10)
    _legend_below(fig, [
        Line2D([], [], marker="o", linestyle="", markersize=7, color="#2a78d6",
               label="repeated stratified CV (shuffled)"),
        Line2D([], [], marker="o", linestyle="", markersize=7, color="#eb6834",
               label="chronological 70/30 split"),
    ], ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "assets"))
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ev = read(runs / "secom_eval.json")
    sw = read(runs / "secom_loop_sweep.json")
    st = read(runs / "secom_stability.json")
    made = []
    if ev:
        made.append(fig_auc(ev, out / "fig_secom_auc.png"))
        made.append(fig_protocol(ev, out / "fig_protocol.png"))
    if sw:
        made.append(fig_sparsity(sw, out / "fig_sparsity.png"))
    if st:
        made.append(fig_stability(st, out / "fig_stability.png"))
    for m in made:
        print(f"wrote {m}")
    if not made:
        print("no run JSONs found; nothing to draw", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
