"""Render the README figures from runs/*.json.

    python scripts/make_figures.py

Nine figures, each one answering a question the tables answer more precisely:

* ``fig_secom_auc.png``  -- who beats whom, and by more than the CI?
* ``fig_sparsity.png``   -- what does selecting fewer sensors cost?
* ``fig_stability.png``  -- how far is top-5 stability from the KPI?
* ``fig_protocol.png``   -- how much of the AUC survives a chronological split?
* ``fig_drift.png``      -- is the process stationary at all?
* ``fig_reversal.png``   -- does the forward-in-time reversal survive the protocol?
* ``fig_premise.png``    -- where does the agent loop help, and where does it hurt?
* ``fig_null_fdr.png``   -- can the drop step tell real causes from permuted labels?
* ``fig_invariance.png`` -- how big a break would the invariance screen even see?

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
    ax.axhline(KPI_AUC, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=2)
    colors = ["#2a78d6", "#eb6834"]
    handles = []
    # one series labels above its markers, the other below, so the two
    # attribution modes' value labels never collide at matched sparsity
    offsets = [(0, 11), (0, -15)]
    for (mode, vals), c, off in zip(sorted(pts.items()), colors, offsets):
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
                        xytext=off, ha="center", fontsize=7.5, color=INK_2)
    ax.set_xscale("log")
    xs_all = [v[0] for vs in pts.values() for v in vs]
    ax.set_xlim(min(xs_all) * 0.72, max(xs_all) * 2.9)
    # the reference lines are labelled where the curves are not: the
    # sparse end sits far below both lines, the dense end runs into them
    ax.text(ax.get_xlim()[0], base["mean"] + 0.002,
            f" rf_all, all sensors  {base['mean']:.3f}", fontsize=8.5,
            color=INK_2, va="bottom", ha="left")
    ax.text(ax.get_xlim()[0], KPI_AUC - 0.002, f" KPI {KPI_AUC:.2f}",
            fontsize=8, color=INK_2, va="top", ha="left")
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
    # a ranker mid-measurement may have bootstrap results but not yet CV ones
    ranked = {k: v for k, v in st["rankers"].items()
              if "bootstrap" in v and "cv_train" in v}
    order = sorted(ranked,
                   key=lambda r: ranked[r]["bootstrap"]["raw"]["pairwise_overlap"])
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(order) + 1.9))
    _grid(ax)
    h = 0.36
    for i, name in enumerate(order):
        r = ranked[name]
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


def fig_reversal(rs, out):
    """Sign stable, magnitude not: the forward-in-time delta at each block count."""
    per = rs["per_block_count"]
    counts = [b for b in sorted(per, key=lambda b: int(b))
              if "agent_vs_rf_all" in per[b]]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    _grid(ax, axis="y")
    xs = range(len(counts))
    for i, b in enumerate(counts):
        d = per[b]["agent_vs_rf_all"]
        ax.plot([i, i], [d["ci_lo"], d["ci_hi"]], color="#2a78d6",
                linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter([i], [d["mean"]], s=46, color="#2a78d6", zorder=4,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.text(i + 0.10, d["mean"], f"{d['mean']:+.3f}", fontsize=8,
                color=INK, va="center", ha="left")
    ax.axhline(0, color=INK_MUTED, linewidth=1.2, zorder=2)
    ax.text(-0.52, -0.008, "no difference", fontsize=8, color=INK_2,
            va="top", ha="left")
    ax.set_xticks(list(xs),
                  [f"{b} blocks\n({per[b]['n_origins']} origins)"
                   for b in counts], fontsize=8.5)
    ax.set_xlim(-0.55, len(counts) - 0.25)
    ax.set_ylabel("paired AUC: agent loop minus\nfull-sensor forest")
    ax.set_title("Forward in time, the sign is stable and the size is not",
                 fontsize=10.5, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_premise(ev, sy, st, out):
    """The sign flip: the loop helps where its premise holds, hurts where it does not.

    Two panels, each with its own single axis -- a paired AUC delta and a
    stability score are different quantities and are never put on one scale.
    """
    d_real = ev["auc"]["paired"]["agent_rf__vs__rf_all"]
    d_syn = sy["auc"]["paired"]["agent_rf__vs__rf_all"]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.4, 3.0))

    labels = ["SECOM\n(real, diffuse signal)",
              "synthetic\n(5 planted causes)"]
    colors = ["#2a78d6", "#eb6834"]

    _grid(axl, axis="x")
    for i, (d, c) in enumerate(zip([d_real, d_syn], colors)):
        axl.barh(i, d["mean"], height=0.5, color=c, zorder=3,
                 edgecolor=SURFACE, linewidth=1.2)
        axl.errorbar(d["mean"], i,
                     xerr=[[d["mean"] - d["ci_lo"]], [d["ci_hi"] - d["mean"]]],
                     fmt="none", ecolor=INK_2, elinewidth=1.5, capsize=3,
                     capthick=1.5, zorder=5)
        # labels sit on the far side of the interval from zero, and the
        # limits are widened to make room rather than letting them collide
        pos = d["mean"] > 0
        axl.text(d["ci_hi"] + 0.004 if pos else d["ci_lo"] - 0.004, i,
                 f"{d['mean']:+.3f}", va="center",
                 ha="left" if pos else "right", fontsize=8.5, color=INK)
    axl.axvline(0, color=INK_MUTED, linewidth=1.1, zorder=4)
    span = max(abs(d["ci_lo"]) for d in (d_real, d_syn)) \
        + max(abs(d["ci_hi"]) for d in (d_real, d_syn))
    axl.set_xlim(min(d_real["ci_lo"], d_syn["ci_lo"]) - 0.32 * span,
                 max(d_real["ci_hi"], d_syn["ci_hi"]) + 0.32 * span)
    axl.set_yticks([0, 1], labels, fontsize=8.5)
    axl.set_ylim(-0.6, 1.6)
    axl.set_xlabel("paired AUC: agent loop minus full-sensor forest")
    axl.set_title("Accuracy: the sign flips", fontsize=10, loc="left", pad=8)

    _grid(axr, axis="x")
    vals = [st["rankers"]["agent"]["bootstrap"]["raw"]["pairwise_overlap"],
            sy["stability"]["agent"]["raw"]["pairwise_overlap"]]
    for i, (v, c) in enumerate(zip(vals, colors)):
        axr.barh(i, v, height=0.5, color=c, zorder=3, edgecolor=SURFACE,
                 linewidth=1.2)
        axr.text(v + 0.012, i, f"{100*v:.1f}%", va="center", ha="left",
                 fontsize=8.5, color=INK)
    axr.axvline(KPI_STAB, color=INK_MUTED, linestyle=(0, (4, 3)),
                linewidth=1.2, zorder=4)
    axr.text(KPI_STAB + 0.01, 1.42, f"KPI {KPI_STAB:.0%}", fontsize=8,
             color=INK_2, va="center", ha="left")
    axr.set_yticks([])            # the left panel already names the rows
    axr.set_ylim(-0.6, 1.6)
    axr.set_xlim(0, 1.12)
    axr.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    axr.set_xlabel("top-5 stability (mean pairwise overlap, bootstrap)")
    axr.set_title("Stability: same story", fontsize=10, loc="left", pad=8)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_drift(dr, out):
    """Fail rate per time block -- the label half of the drift diagnosis."""
    blocks = dr["label_drift"]["per_block"]
    overall = sum(b["n_fail"] for b in blocks) / sum(b["n"] for b in blocks)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    _grid(ax, axis="y")
    xs = [b["block"] for b in blocks]
    ys = [b["fail_rate"] for b in blocks]
    ax.bar(xs, ys, width=0.62, color="#2a78d6", zorder=3, edgecolor=SURFACE,
           linewidth=1.2)
    for b in blocks:
        ax.text(b["block"], b["fail_rate"] + 0.004,
                f"{100*b['fail_rate']:.1f}%\n{b['n_fail']}/{b['n']}",
                ha="center", va="bottom", fontsize=8, color=INK,
                linespacing=1.35)
    ax.axhline(overall, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=4)
    ax.text(-0.95, overall, f"overall {100*overall:.1f}%", fontsize=8,
            color=INK_2, va="bottom", ha="left")
    ax.set_ylim(0, max(ys) * 1.42)
    ax.set_xlim(-1.05, len(blocks) - 0.4)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    ax.set_xticks(xs, [f"block {i}\n(earliest first)" if i == 0 else f"block {i}"
                       for i in xs], fontsize=8.5)
    ax.set_ylabel("fail rate")
    adv = dr["adversarial"]["auc"]["mean"]
    ax.set_title(f"The process is non-stationary: fail rate moves across the "
                 f"campaign\n(and era is predictable from the sensors at "
                 f"AUC {adv:.3f})", fontsize=10, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_null_fdr(nf, out):
    """Null vs real: the statistic the drop step thresholds, side by side.

    The question the figure has to answer at a glance is whether the loop's own
    confidence knows the difference between real labels and permuted ones. Two
    histograms of the same statistic and one threshold line answer it; the
    verdict is in the title so the figure survives being screenshotted out of
    context.
    """
    null = [r["max_stability"] for r in nf["records"] if r["permuted"]]
    real = [r["max_stability"] for r in nf["records"] if not r["permuted"]]
    tau = nf["thresholds"]["alpha_0.05"]
    thr = nf["protocol"]["agent_cfg"]["stability_min"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    _grid(ax, axis="y")
    lo = min(min(null), min(real), thr) - 0.05
    bins = [lo + i * (1.02 - lo) / 22 for i in range(23)]
    ax.hist(null, bins=bins, color="#eb6834", alpha=0.72, zorder=3,
            edgecolor=SURFACE, linewidth=0.8, density=True,
            label=f"permuted labels, no causes exist (n={len(null)})")
    ax.hist(real, bins=bins, color="#2a78d6", alpha=0.72, zorder=3,
            edgecolor=SURFACE, linewidth=0.8, density=True,
            label=f"real labels (n={len(real)})")
    ax.axvline(thr, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.3,
               zorder=5)
    ax.text(thr, ax.get_ylim()[1] * 0.97, f" drop threshold {thr:g}",
            fontsize=8, color=INK_2, va="top", ha="left")
    ax.axvline(tau, color=INK, linestyle=(0, (1, 2)), linewidth=1.4, zorder=5)
    ax.text(tau, ax.get_ylim()[1] * 0.72,
            f" null-calibrated\n tau(0.05) = {tau:.2f}", fontsize=8,
            color=INK, va="top", ha="left", linespacing=1.4)
    ax.set_xlabel("largest bootstrap support any reported suspect reached")
    ax.set_ylabel("density")
    sep = nf["separation"]["prob_real_max_exceeds_null_max"]
    fdr = nf["null_fdr"]["fdr_given_nonempty"]
    absten = nf["null"]["abstention_rate"]
    ax.set_title(
        f"The drop threshold does not separate real causes from noise\n"
        f"false-discovery rate on permuted labels {fdr:.0%}, abstention "
        f"{absten:.0%}, P(real > null) = {sep:.2f}",
        fontsize=10, loc="left", pad=10)
    _legend_below(fig, ax.get_legend_handles_labels()[0], ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_invariance(iv, out):
    """The power curve, with SECOM's real signal strengths on a shared axis.

    A null invariance result is only informative next to the effect size the
    test could have caught, so the two share an x axis: above, the detection
    rate against breaks of known size; below, what the dataset's associated
    sensors actually offer. The gap between the two panels *is* the finding, and
    the one sensor that sits far enough right to be convicted is labelled.
    """
    ladder = iv["power"]["ladder"]
    fam = iv["power"]["n_family"]
    xs = [r["target_block0_auc"] for r in ladder]
    assoc = [r for r in iv["per_sensor"] if r.get("associated")]
    folded = sorted(0.5 + abs(r["pooled_auc"] - 0.5) for r in assoc)
    broke = [r for r in assoc if not r.get("invariant")]
    peaks = {r["name"]: max(v for v in r["block_auc"] if v is not None)
             for r in broke}
    hi = max(xs + list(peaks.values())) + 0.02
    lo = min(xs + folded) - 0.02

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(6.8, 4.3), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.16})
    _grid(ax, axis="y")
    ax.plot(xs, [r["power_at_alpha"] for r in ladder], marker="o", ms=4.5,
            color="#2a78d6", linewidth=1.8, zorder=4,
            label="detected at p < 0.05")
    ax.plot(xs, [r["power_at_alpha_over_family"] for r in ladder], marker="s",
            ms=4.0, color="#eb6834", linewidth=1.8, zorder=4,
            label=f"detected at p < 0.05/{fam} (strictest BH level)")
    for r in ladder:
        ax.text(r["target_block0_auc"], r["power_at_alpha"] + 0.05,
                f"{100*r['power_at_alpha']:.0f}%", ha="center", fontsize=7.6,
                color=INK_2)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    ax.set_ylabel("detection rate")
    ax.set_title("The invariance screen cannot see a break the size SECOM's "
                 "sensors could carry\nso 'not rejected' means 'not testable', "
                 "not 'invariant'", fontsize=10, loc="left", pad=10)

    # lower strip: what the dataset actually offers
    for v in folded:
        bx.plot([v, v], [0.52, 0.95], color=INK_MUTED, linewidth=1.1,
                solid_capstyle="butt", zorder=4)
    bx.text(folded[0] - 0.004, 1.25,
            f"pooled association of the {len(folded)} sensors SECOM offers",
            ha="left", va="bottom", fontsize=8, color=INK_2)
    for name, peak in peaks.items():
        bx.plot([peak], [0.73], marker="v", ms=7, color="#1baf7a", zorder=6)
        bx.annotate(f"{name}: peak period {peak:.2f}\nthe one sensor rejected",
                    xy=(peak, 0.95), xytext=(peak, 1.9),
                    ha="center", va="bottom", fontsize=8, color=INK,
                    linespacing=1.4,
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9))
    bx.set_ylim(0, 2.6)
    bx.set_xlim(lo, hi)
    bx.set_yticks([])
    for side in ("left", "top", "right"):
        bx.spines[side].set_visible(False)
    bx.set_xlabel("association within one production period "
                  "(AUC; 0.5 = none)", labelpad=6)
    fig.legend(handles=ax.get_legend_handles_labels()[0], loc="lower center",
               ncol=2, frameon=False, fontsize=8.5, labelcolor=INK_2,
               handletextpad=0.4, columnspacing=1.8, borderpad=0,
               bbox_to_anchor=(0.5, -0.075))
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
    dr = read(runs / "drift.json")
    if dr:
        made.append(fig_drift(dr, out / "fig_drift.png"))
    rs = read(runs / "rolling_sweep.json")
    if rs:
        made.append(fig_reversal(rs, out / "fig_reversal.png"))
    sy = read(runs / "synthetic.json")
    if ev and sy and st and "agent" in st.get("rankers", {}):
        made.append(fig_premise(ev, sy, st, out / "fig_premise.png"))
    nf = read(runs / "null_fdr.json")
    if nf and nf.get("records"):
        made.append(fig_null_fdr(nf, out / "fig_null_fdr.png"))
    iv = read(runs / "invariance.json")
    if iv and (iv.get("power") or {}).get("ladder"):
        made.append(fig_invariance(iv, out / "fig_invariance.png"))
    for m in made:
        print(f"wrote {m}")
    if not made:
        print("no run JSONs found; nothing to draw", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
