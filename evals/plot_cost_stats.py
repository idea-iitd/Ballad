"""
plot_cost_stats.py

Plot version of build_cost_table.py: reads proc/<problem>/cost_stats.json files
(produced by model_cost_stats.py) and renders a grid of bar plots — one subplot
per problem — showing mean total_cost (USD) per model with std as error bars.

Each subplot has its own y-axis (USD ranges differ wildly across problems).

Usage:
    python evals/plot_cost_stats.py
    python evals/plot_cost_stats.py --proc_dir proc \\
        --problems tsp_constructive cvrp_aco bpp_online bpp_offline_aco ged \\
        --out proc/cost_stats_bars.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Problem & model configuration (kept in lock-step with build_cost_table.py)

# ── Problem & model configuration (kept in lock-step with build_cost_table.py)

PROBLEMS = [
    ("TSP",         "tsp_constructive"),
    ("CVRP",        "cvrp_aco"),
    ("Online BP",   "bpp_online"),
    ("Offline BP",  "bpp_offline_aco"),
    ("GED", "ged")
]

# (display label, candidate keys to look up in cost_stats.json — first hit wins)
ROWS = [
    ("Gemini 2.5 Flash-Lite", ["gemini-2.5-flash-lite"]),
    ("Gemini 2.5 Flash",      ["gemini-2.5-flash"]),
    ("Ballad",      ["ballad"]),
]

MODEL_ALIASES = {
} #use if required
 
# Vivid, modern palette (Tableau-10 inspired) — saturated enough to read well
# on slides while staying paper-friendly.
ROW_COLORS = {
    "Gemini 2.5 Flash-Lite": "#4E79A7",  # deep blue
    "Gemini 2.5 Flash":      "#EDC948",  # teal
    "Ballad":      "#E15759",  # coral red
}

# Single-density hatch patterns — distinct shapes, redundant with colour.
ROW_HATCHES = {
    "Gemini 2.5 Flash-Lite": "+++",
    "Gemini 2.5 Flash":      "\\\\\\",
    "Ballad":                "ooo",
}


def resolve_model(prob_stats: dict, candidates: list) -> str | None:
    seen = set()
    for name in candidates:
        for cand in [name] + MODEL_ALIASES.get(name, []):
            if cand in seen:
                continue
            seen.add(cand)
            if cand in prob_stats:
                return cand
    return None


def load_problem_costs(proc_dir: Path, prob: str) -> dict:
    """Return {row_label: (mean, std, n_runs)} for one problem."""
    out = {}
    json_path = proc_dir / prob / "cost_stats.json"
    if not json_path.exists():
        return out
    data = json.loads(json_path.read_text())
    models = data.get("models", {})
    for label, candidates in ROWS:
        key = resolve_model(models, candidates)
        if key is None:
            continue
        tc = models[key].get("total_cost", {})
        out[label] = (
            float(tc.get("mean", 0.0)),
            float(tc.get("std", 0.0)),
            int(models[key].get("n_runs", 0)),
        )
    return out


def _autoscale_ylim(ax, means, stds, headroom: float = 1.55):
    """Add headroom above the tallest bar+error so the in-panel legend has room."""
    tops = [m + s for m, s in zip(means, stds) if m is not None]
    if not tops:
        return
    top = max(tops)
    if top <= 0:
        ax.set_ylim(0, 1)
        return
    ax.set_ylim(0, top * headroom)


def plot_subplot(ax, title, costs, panel_letter: str = "", show_ylabel: bool = True):
    """
    costs: dict[label -> (mean, std, n_runs)]; preserves ROWS order.
    panel_letter: one of "(a)", "(b)", ...; rendered as part of the bottom caption.
    show_ylabel: render the "Total cost per run (USD)" label on this axes.
    """
    caption = f"{panel_letter} {title}".strip() if panel_letter else title
    labels, means, stds = [], [], []
    for label, _ in ROWS:
        if label not in costs:
            continue
        mean, std, _ = costs[label]
        labels.append(label)
        means.append(mean)
        stds.append(std)

    if not labels:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{caption}  (no data)", fontweight="bold", labelpad=10)
        return

    x = np.arange(len(labels))
    bar_handles = []

    for xi, lbl, m, s in zip(x, labels, means, stds):
        bar = ax.bar(
            xi, m,
            yerr=s,
            color=ROW_COLORS.get(lbl, "#888888"),
            edgecolor="#222222",
            linewidth=0.6,
            width=0.72,
            hatch=ROW_HATCHES.get(lbl, ""),
            capsize=3,
            error_kw=dict(ecolor="#222222", elinewidth=0.9),
            label=lbl,
            zorder=2,
        )
        bar_handles.append(bar)

    # Value labels above each bar
    for xi, m, s in zip(x, means, stds):
        ax.annotate(
            f"${m:.3g}",
            xy=(xi, m + s),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold",
            color="#222222",
        )

    # Hide model names on x-axis — the shared figure legend covers identification.
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(labels))
    ax.tick_params(axis="x", which="both", length=0)
    if show_ylabel:
        ax.set_ylabel("Total cost per run (USD)", fontweight="bold")
    else:
        ax.set_ylabel("")
    # Panel caption (e.g. "(a) TSP") below the empty x-axis, centred.
    ax.set_xlabel(caption, fontweight="bold", labelpad=10, fontsize=18)
    # Y-axis numerical tick labels stay regular (not bold) per spec.
    for tick in ax.get_yticklabels():
        tick.set_fontweight("normal")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _autoscale_ylim(ax, means, stds, headroom=1.18)


def make_figure(proc_dir: Path, problems: list, out_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Times", "DejaVu Serif"],
        "font.size": 15,
        "font.weight": "bold",
        "axes.titlesize": 19,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 17,
        "legend.fontsize": 13,
        "figure.titlesize": 20,
    })

    n = len(problems)
    n_cols = n
    n_rows = 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 5.0 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i, (header, prob_dir_name) in enumerate(problems):
        costs = load_problem_costs(proc_dir, prob_dir_name)
        letter = f"({chr(ord('a') + i)})"
        plot_subplot(
            axes_flat[i], header, costs,
            panel_letter=letter,
            show_ylabel=(i == 0),   # only the leftmost panel labels the y-axis
        )

    # Hide any unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # One shared, horizontal legend at the top centre of the figure.
    legend_handles = [
        mpatches.Patch(
            facecolor=ROW_COLORS[label],
            edgecolor="#222222",
            linewidth=0.6,
            hatch=ROW_HATCHES.get(label, ""),
            label=label,
        )
        for label, _ in ROWS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(legend_handles),
        frameon=False,
        prop={"weight": "bold", "size": 21},
        handlelength=2.7,
        handleheight=2.1,
        columnspacing=1.8,
    )

    # Leave room at the top for the legend.
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[+] Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a grid of cost bar charts (one subplot per problem)."
    )
    parser.add_argument("--proc_dir", default="proc",
                        help="Processed results root (default: proc).")
    parser.add_argument("--problems", nargs="+", default=None,
                        help="Subset of problem dir names. Default: all from PROBLEMS.")
    parser.add_argument("--out", default=None,
                        help="Output PDF/PNG path (default: <proc_dir>/cost_stats/cost_bars.pdf).")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    proc_dir = root / args.proc_dir

    if args.problems:
        wanted = set(args.problems)
        problems = [(hdr, name) for hdr, name in PROBLEMS if name in wanted]
        missing = wanted - {name for _, name in problems}
        for m in missing:
            print(f"[!] Unknown problem '{m}' — not in PROBLEMS list.")
    else:
        problems = PROBLEMS

    out_path = (
        Path(args.out)
        if args.out
        else proc_dir / "cost_stats" / "cost_bars.pdf"
    )

    make_figure(proc_dir, problems, out_path)


if __name__ == "__main__":
    main()
