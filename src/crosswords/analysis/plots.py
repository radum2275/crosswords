# Figures for paper/error_analysis.tex, drawn from the CSVs error_analysis.py
# writes. Static PDFs for print: every line series carries a marker shape and a
# direct label in addition to its color, since two of the four palette slots are
# below 3:1 contrast on white.
#
# Usage:
#   PYTHONPATH=src python -m crosswords.analysis.plots

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from crosswords.analysis.error_analysis import DATASETS, DATASET_LABEL, ERROR_LABEL, ERROR_TYPES
from crosswords.analysis.rescore import MODEL_ORDER

# Categorical slots 1-5 in fixed order (validated: adjacent CVD dE >= 9.1). Colors
# follow the model, not its position, so they stay put when a model is absent.
MODEL_COLOR = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MODEL_MARKER = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v"]))
# Correct + five error types: categorical slots 1-6.
ERROR_COLOR = dict(zip(["correct"] + ERROR_TYPES,
                       ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SHORT = dict(zip(MODEL_ORDER, ["Granite", "Llama", "gpt-oss", "Claude", "LLMic"]))

LADDER = [("v1", 0, "None"), ("v2", 0, "Len"), ("v3", 1, "k=1"), ("v3", 2, "k=2"),
          ("v3", 3, "k=3"), ("v3", 4, "k=4")]
TEXT_W = 5.5  # ICLR text width, inches


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "lines.markersize": 4, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def pct_axis(ax, top: float = 1.04) -> None:
    ax.set_ylim(0, top)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def label_line_ends(ax, ends: dict, x: float, min_gap: float = 0.055) -> None:
    """Direct labels at the right end of each line, nudged apart to avoid overlap."""
    ax.set_xlim(right=x + 1.25)
    ax.spines["bottom"].set_bounds(ax.get_xlim()[0], x + 0.3)
    items = sorted(((m, y) for m, y in ends.items() if not np.isnan(y)), key=lambda kv: kv[1])
    placed = []
    for m, y in items:
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
        ax.annotate(SHORT[m], (x, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.5, color=INK_2,
                    annotation_clip=False)


def fig_ladder(acc: pd.DataFrame, out: str) -> None:
    """Accuracy along the hint ladder, no-diacritic (solid) vs strict (faint)."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.4), sharey=True)
    xs = np.arange(len(LADDER))
    for ax, ds in zip(axes, DATASETS):
        ends = {}
        for m in MODEL_ORDER:
            ys, ys_s = [], []
            for v, k, _ in LADDER:
                r = acc[(acc.dataset == ds) & (acc.model == m) & (acc.version == v) & (acc.k == k)]
                ys.append(r.acc.iloc[0] if len(r) else np.nan)
                ys_s.append(r.acc_strict.iloc[0] if len(r) else np.nan)
            c = MODEL_COLOR[m]
            ax.plot(xs, ys_s, color=c, lw=0.8, ls=(0, (2, 2)), alpha=0.6)
            ax.plot(xs, ys, color=c, marker=MODEL_MARKER[m], label=m,
                    markeredgecolor="white", markeredgewidth=0.6, markersize=5)
            ends[m] = ys[-1]
        label_line_ends(ax, ends, xs[-1])
        ax.set_xticks(xs, [lab for *_, lab in LADDER])
        ax.set_title(DATASET_LABEL[ds])
        ax.set_xlabel("Hint")
        pct_axis(ax)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK_2, lw=0.8, ls=(0, (2, 2))))
    labels.append("strict (diacritics count)")
    # Legend in two rows so it is no wider than the figure; a wider legend makes
    # bbox="tight" grow the canvas and the panels shrink on the page.
    fig.legend(handles, labels, loc="upper center", ncol=(len(handles) + 1) // 2, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=1.0)
    fig.savefig(out)
    plt.close(fig)


def fig_strata(st: pd.DataFrame, acc: pd.DataFrame, out: str) -> None:
    """v3 accuracy by how many answer letters the hint leaves hidden."""
    panels = [("all", "All items"), ("revealed", r"Answer fully revealed ($|a|\leq k$)"),
              ("missing_1_2", "1-2 letters hidden"), ("missing_3plus", "3+ letters hidden")]
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_W, 3.3), sharey=True, sharex=True)
    ks = [1, 2, 3, 4]
    for i, ds in enumerate(DATASETS):
        for j, (s, title) in enumerate(panels):
            ax = axes[i, j]
            for m in MODEL_ORDER:
                if s == "all":
                    d = acc[(acc.dataset == ds) & (acc.model == m) & (acc.version == "v3")]
                else:
                    d = st[(st.dataset == ds) & (st.model == m) & (st.version == "v3") & (st.stratum == s)]
                y = [d[d.k == k].acc.iloc[0] if (d.k == k).any() else np.nan for k in ks]
                ax.plot(ks, y, color=MODEL_COLOR[m], marker=MODEL_MARKER[m], label=m,
                        markeredgecolor="white", markeredgewidth=0.5)
            if s == "revealed":
                ax.set_xlim(0.6, 4.4)
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            ax.set_xticks(ks, [f"k={k}" for k in ks])
            pct_axis(ax)
        axes[i, 0].set_ylabel(f"{DATASET_LABEL[ds]}\naccuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)


def fig_candidates(cand: pd.DataFrame, out: str) -> None:
    """v3 accuracy (pooled over k) against how many lexicon words fit the hint."""
    order = ["1", "2-3", "4-10", "11-30", "31-100", ">100"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        d = cand[cand.dataset == ds]
        ends = {}
        for m in MODEL_ORDER:
            g = d[d.model == m].groupby("cand_bin").apply(
                lambda x: np.average(x.acc, weights=x.n), include_groups=False).reindex(order)
            ax.plot(range(len(order)), g.values, color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                    label=m, markeredgecolor="white", markeredgewidth=0.5)
            ends[m] = g.values[-1]
        label_line_ends(ax, ends, len(order) - 1)
        ax.set_xticks(range(len(order)), order, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_title(DATASET_LABEL[ds])
        pct_axis(ax)
    axes[0].set_ylabel("Accuracy (v3, $k$=1-4 pooled,\nanswer not fully revealed)")
    fig.supxlabel("Crossword words fitting the length + prefix hint", fontsize=8)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out)
    plt.close(fig)


def fig_length_heatmap(bl: pd.DataFrame, out: str) -> None:
    """Accuracy by answer length x hint level, for the two strongest models."""
    models = ["gpt-oss-120B", "Claude Opus 4.8"]
    cmap = LinearSegmentedColormap.from_list("blue", ["#f7f9fc"] + BLUE_RAMP)
    lengths = list(range(3, 11))
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W, 3.6), sharex=True, sharey=True)
    for i, ds in enumerate(DATASETS):
        for j, m in enumerate(models):
            ax = axes[i, j]
            grid = np.full((len(lengths), len(LADDER)), np.nan)
            for c, (v, k, _) in enumerate(LADDER):
                d = bl[(bl.dataset == ds) & (bl.model == m) & (bl.version == v) & (bl.k == k)]
                for r, n in enumerate(lengths):
                    x = d[d.length_bin == n]
                    if len(x):
                        grid[r, c] = x.acc.iloc[0]
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
            for r in range(len(lengths)):
                for c in range(len(LADDER)):
                    if not np.isnan(grid[r, c]):
                        v = grid[r, c]
                        ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5,
                                color="white" if v > 0.55 else INK)
            # Cells where the hint already spells out the whole answer.
            for c, (v, k, _) in enumerate(LADDER):
                for r, n in enumerate(lengths):
                    if v == "v3" and n <= k:
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                                   ec=INK, lw=0.7, ls=(0, (1.5, 1))))
            ax.grid(False)
            ax.set_xticks(range(len(LADDER)), [lab for *_, lab in LADDER])
            ax.set_yticks(range(len(lengths)), [str(n) if n < 10 else "10+" for n in lengths])
            if i == 0:
                ax.set_title(m)
            if j == 0:
                ax.set_ylabel(f"{DATASET_LABEL[ds]}\nanswer length")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_errors(err: pd.DataFrame, acc: pd.DataFrame, out: str) -> None:
    """Share of all items that are correct or fall in each error type."""
    conds = LADDER[1:]  # v1 has no hint to violate
    cats = ["correct"] + ERROR_TYPES
    labels = {"correct": "Correct", **ERROR_LABEL}
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(TEXT_W, 3.6), sharey=True, sharex=True)
    for i, ds in enumerate(DATASETS):
        for j, m in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            bottom = np.zeros(len(conds))
            for cat in cats:
                vals = []
                for v, k, _ in conds:
                    if cat == "correct":
                        r = acc[(acc.dataset == ds) & (acc.model == m) & (acc.version == v) & (acc.k == k)]
                        vals.append(r.acc.iloc[0] if len(r) else 0)
                    else:
                        r = err[(err.dataset == ds) & (err.model == m) & (err.version == v) & (err.k == k)]
                        vals.append(r[cat].iloc[0] / r.n.iloc[0] if len(r) else 0)
                vals = np.array(vals)
                ax.bar(range(len(conds)), vals, bottom=bottom, width=0.78, color=ERROR_COLOR[cat],
                       edgecolor="white", linewidth=0.6, label=labels[cat])
                bottom += vals
            missing_run = [not ((acc.dataset == ds) & (acc.model == m) & (acc.version == v) & (acc.k == k)).any()
                           for v, k, _ in conds]
            for c, miss in enumerate(missing_run):
                if miss:
                    ax.text(c, 0.5, "n/a", ha="center", va="center", fontsize=6, color=INK_2, rotation=90)
            ax.set_xticks(range(len(conds)), [lab for *_, lab in conds], rotation=45)
            ax.grid(False)
            pct_axis(ax)
            if i == 0:
                ax.set_title(m.replace("-small", ""), fontsize=7.5)
        axes[i, 0].set_ylabel(f"{DATASET_LABEL[ds]}\nshare of items")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    # Three columns keep the legend within the figure width (see fig_ladder).
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    plt.close(fig)


def fig_agreement(ag: pd.DataFrame, out: str) -> None:
    """How many of the four models solve each item, per condition."""
    conds = [(v, k, lab) for v, k, lab in LADDER]
    # 0 models = neutral gray, then the blue ramp from step 250 up (ordinal).
    ramp = ["#e9e8e4", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.0), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        d = ag[ag.dataset == ds]
        present = [(v, k, lab) for v, k, lab in conds if ((d.version == v) & (d.k == k)).any()]
        bottom = np.zeros(len(present))
        n_models = int(d["n_models"].max())
        if n_models + 1 < len(ramp):
            # Keep the darkest step for "all models".
            ramp = ramp[:n_models] + ramp[-1:]
        for n in range(n_models + 1):
            vals = []
            for v, k, _ in present:
                r = d[(d.version == v) & (d.k == k)].iloc[0]
                tot = sum(r[f"solved_by_{i}"] for i in range(n_models + 1))
                vals.append(r[f"solved_by_{n}"] / tot)
            vals = np.array(vals)
            ax.bar(range(len(present)), vals, bottom=bottom, width=0.78, color=ramp[n],
                   edgecolor="white", linewidth=0.6, label=f"{n}")
            bottom += vals
        ax.set_xticks(range(len(present)), [lab for *_, lab in present])
        ax.set_title(DATASET_LABEL[ds])
        ax.grid(False)
        pct_axis(ax)
    axes[0].set_ylabel("Share of items")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, title="Models solving the item", loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False, title_fontsize=7)
    fig.savefig(out)
    plt.close(fig)


def main(analysis_dir: str, fig_dir: str) -> None:
    setup()
    os.makedirs(fig_dir, exist_ok=True)
    r = lambda n: pd.read_csv(os.path.join(analysis_dir, f"{n}.csv"), dtype={"cand_bin": str})
    acc, st = r("accuracy"), r("strata")
    # Only draw models that have results (see error_analysis.main).
    global MODEL_ORDER
    MODEL_ORDER = [m for m in MODEL_ORDER if m in set(acc["model"])]
    fig_ladder(acc, os.path.join(fig_dir, "accuracy_ladder.pdf"))
    fig_strata(st, acc, os.path.join(fig_dir, "strata.pdf"))
    fig_candidates(r("candidates"), os.path.join(fig_dir, "candidates.pdf"))
    fig_length_heatmap(r("by_length"), os.path.join(fig_dir, "length_heatmap.pdf"))
    fig_errors(r("errors"), acc, os.path.join(fig_dir, "errors.pdf"))
    fig_agreement(r("agreement"), os.path.join(fig_dir, "agreement.pdf"))
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", default="data/results/analysis")
    parser.add_argument("--fig_dir", default="paper/figures")
    args = parser.parse_args()
    main(args.analysis_dir, args.fig_dir)
