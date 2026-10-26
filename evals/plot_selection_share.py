"""
plot_selection_share.py

Stacked-area plot of which LLM Ballad routes each call to, as the search
progresses.  One subplot per problem; one shared horizontal legend at the top.

Data source per (problem, run):
    results/<problem>/<ballad-folder>/code_dict_run{N}.pkl
        — field 'gen_model' on every call entry tells which LLM was selected
          by the router for that call.

Run discovery mirrors build_table2.py / model_cost_stats.py: per-(problem,model)
sorted runs from inference_results/<problem>/<model>/test_best_metric_run*.pkl.

Usage:
    python evals/plot_selection_share.py
    python evals/plot_selection_share.py --window 50
    python evals/plot_selection_share.py --problems tsp_constructive cvrp_aco

[DISCLAIMER] Inference results are currently used only to extract run numbers. 
Since these choices are determined during training, the script can be modified to use run numbers directly 
and avoid relying on inference outputs.
"""

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Same problem set / palette / hatches as the cost-bars figure ───────────

PROBLEMS = [
    ("TSP",         "tsp_constructive"),
    ("CVRP",        "cvrp_aco"),
    ("Online BP",   "bpp_online"),
    ("Offline BP",  "bpp_offline_aco"),
    ("GED", "ged")
]

# gen_model strings in the logs are compact IDs.  Canonicalise to the names
# everyone else uses.  Add new ones here as needed.
KEY_TO_CANONICAL = {
    "gemini_flash_lite": "gemini-2.5-flash-lite",
    "gemini_flash":      "gemini-2.5-flash",
    "gemma_3_12b":       "gemma-3-12b-it",
    "gemma_3_27b":       "gemma-3-27b-it",
}

# Stack ordering: cheapest at the bottom, most expensive at the top.  The
# reader's eye reads "how much area is in the cheap region" → that's the win.
ARM_ORDER = [
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

ARM_DISPLAY = {
    "gemma-3-12b-it":        "Gemma 3 12B",
    "gemma-3-27b-it":        "Gemma 3 27B",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "gemini-2.5-flash":      "Gemini 2.5 Flash",
}


ARM_COLORS = {
    "gemma-3-12b-it":        "#009E73",  # bluish green (Okabe–Ito)
    "gemma-3-27b-it":        "#CC79A7",  # reddish purple (Okabe–Ito)
    "gemini-2.5-flash-lite": "#4E79A7",
    "gemini-2.5-flash":      "#EDC948",
}


ARM_HATCHES = {
    "gemma-3-12b-it":        "...",   # fine dots (light texture)
    "gemma-3-27b-it":        "---",   # horizontal lines (clean, very distinct)
    "gemini-2.5-flash-lite": "+++",   # consistent
    "gemini-2.5-flash":      "\\\\\\",  # consistent
}

# Folder name under results/ that holds Ballad runs.
BALLAD_FOLDERS = ["ballad"] 


# ── Discovery (mirrors build_table2.detect_runs_for) ───────────────────────

def detect_runs_for(problem: str, model_folder: str, inference_root: Path) -> list:
    pattern = re.compile(r"^test_best_metric_run(\d+)\.pkl$")
    runs = set()
    d = inference_root / problem / model_folder
    if d.is_dir():
        for f in d.iterdir():
            m = pattern.match(f.name)
            if m:
                runs.add(int(m.group(1)))
    return sorted(runs)


def resolve_model_folder(problem: str, results_root: Path,
                         inference_root: Path) -> tuple | None:
    """Pick the first BALLAD_FOLDERS entry that has both run logs and inference
    test_best_metric pkls for this problem.  Returns (folder_name, [run_nums])."""
    for folder in BALLAD_FOLDERS:
        runs = detect_runs_for(problem, folder, inference_root)
        log_dir = results_root / problem / folder
        if runs and log_dir.is_dir():
            valid = [r for r in runs if (log_dir / f"code_dict_run{r}.pkl").exists()]
            if valid:
                return folder, valid
    return None


# ── Data extraction ────────────────────────────────────────────────────────

def load_run_selections(log_path: Path) -> list:
    """Return list of canonical model ids per call (call 0 seed dropped)."""
    with open(log_path, "rb") as f:
        log = pickle.load(f)
    seq = []
    for idx, entry in enumerate(log):
        if idx == 0:
            continue                       # seed/placeholder
        m = entry.get("gen_model")
        if not m:
            continue
        seq.append(KEY_TO_CANONICAL.get(m, m))
    return seq


def rolling_share(seq: list, arms: list, window: int) -> np.ndarray:
    """Return shares[t, a] = fraction of arm a in window ending at call t."""
    n = len(seq)
    onehot = np.zeros((n, len(arms)), dtype=float)
    arm_idx = {a: i for i, a in enumerate(arms)}
    for t, m in enumerate(seq):
        if m in arm_idx:
            onehot[t, arm_idx[m]] = 1.0

    # Rolling sum via cumulative trick (efficient and exact)
    cs = np.cumsum(onehot, axis=0)
    pad = np.zeros((1, len(arms)))
    cs_p = np.vstack([pad, cs])
    win_count = np.minimum(np.arange(1, n + 1), window).astype(float)
    win_lo    = np.maximum(np.arange(n) - window + 1, 0)
    rolling   = (cs_p[np.arange(n) + 1] - cs_p[win_lo]) / win_count[:, None]
    return rolling                                   # shape (n, n_arms)


def shares_for_problem(problem: str, results_root: Path, inference_root: Path,
                       window: int) -> dict:
    res = resolve_model_folder(problem, results_root, inference_root)
    if res is None:
        return {}
    folder, runs = res
    log_dir = results_root / problem / folder

    per_run = []
    n_max = 0
    for r in runs:
        seq = load_run_selections(log_dir / f"code_dict_run{r}.pkl")
        if not seq:
            continue
        per_run.append(seq)
        n_max = max(n_max, len(seq))

    if not per_run:
        return {}

    # Compute rolling share per run, then mean across runs (NaN-safe at tail).
    stacks = []
    for seq in per_run:
        share = rolling_share(seq, ARM_ORDER, window)        # (n_run, K)
        if share.shape[0] < n_max:
            pad = np.full((n_max - share.shape[0], share.shape[1]), np.nan)
            share = np.vstack([share, pad])
        stacks.append(share)
    stack = np.stack(stacks, axis=0)                         # (R, n_max, K)
    mean_share = np.nanmean(stack, axis=0)                   # (n_max, K)

    return {
        "folder": folder,
        "runs":   runs,
        "x":      np.arange(1, n_max + 1),
        "share":  mean_share,
        "n_runs": len(per_run),
    }


# ── Plot ───────────────────────────────────────────────────────────────────

def plot_subplot(ax, title, panel_letter, data, show_ylabel: bool):
    caption = f"{panel_letter} {title}".strip()

    if not data:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{caption}  (no data)", fontweight="bold", labelpad=10)
        return

    x = data["x"]
    share = data["share"]                  # (T, K)
    K = share.shape[1]

    # stackplot bottom→top in ARM_ORDER
    components = [share[:, i] for i in range(K)]
    colors = [ARM_COLORS[a] for a in ARM_ORDER]

    polys = ax.stackplot(
        x, components,
        colors=colors,
        edgecolor="#232323",  
        linewidth=0.3,
        alpha=0.95
    )
    for poly, arm in zip(polys, ARM_ORDER):
        poly.set_hatch(ARM_HATCHES[arm])


    ax.axhline(
        1.0 / K,
        color="#D62728",
        linestyle="--",
        linewidth=2.5,
        alpha=0.9,
        zorder=5
    )

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(caption, fontweight="bold", labelpad=10, fontsize=22)
    if show_ylabel:
        ax.set_ylabel("Selection share (rolling)", fontweight="bold", fontsize=20)
    else:
        ax.set_ylabel("")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("normal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def make_figure(results_root: Path, inference_root: Path, problems: list,
                window: int, out_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Times", "DejaVu Serif"],
        "font.size": 17,
        "font.weight": "bold",
        "axes.labelsize": 19,
        "axes.labelweight": "bold",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    n = len(problems)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5.2), squeeze=False)
    axes_flat = axes.ravel()

    for i, (header, prob) in enumerate(problems):
        data = shares_for_problem(prob, results_root, inference_root, window)
        letter = f"({chr(ord('a') + i)})"
        plot_subplot(axes_flat[i], header, letter, data, show_ylabel=(i == 0))
        if data:
            print(f"[+] {prob}: folder={data['folder']}  "
                  f"runs={data['runs']}  T={len(data['x'])}")
        else:
            print(f"[!] {prob}: no Ballad data found")

    edge_col = "#232323"

    legend_handles = [
        mpatches.Patch(
            facecolor=ARM_COLORS[a],
            edgecolor=edge_col,
            linewidth=0.6,
            hatch=ARM_HATCHES[a],
            label=ARM_DISPLAY[a]
        )
        for a in ARM_ORDER
    ]
    fig.legend(handles=legend_handles,
               loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=len(legend_handles), frameon=False,
               prop={"weight": "bold", "size": 21},
               handlelength=2.7, handleheight=2.1, columnspacing=1.8)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[+] Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Stacked-area selection-share plot for Ballad."
    )
    parser.add_argument("--results_dir", default="results",
                        help="Root results directory (default: results).")
    parser.add_argument("--inference_dir", default="inference_results",
                        help="Inference results root (default: inference_results).")
    parser.add_argument("--problems", nargs="+", default=None,
                        help="Subset of problem dir names. Default: all from PROBLEMS.")
    parser.add_argument("--window", type=int, default=100,
                        help="Rolling window length (calls). Default 100.")
    parser.add_argument("--out", default=None,
                        help="Output PDF path "
                             "(default: proc/cost_stats/selection_share.pdf).")
    args = parser.parse_args()

    root           = Path(__file__).resolve().parent.parent
    results_root   = root / args.results_dir
    inference_root = root / args.inference_dir

    if args.problems:
        wanted = set(args.problems)
        problems = [(hdr, p) for hdr, p in PROBLEMS if p in wanted]
    else:
        problems = PROBLEMS

    out_path = (
        Path(args.out) if args.out
        else root / "proc" / "cost_stats" / "selection_share.pdf"
    )

    make_figure(results_root, inference_root, problems, args.window, out_path)


if __name__ == "__main__":
    main()
