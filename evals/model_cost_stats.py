"""
model_cost_stats.py

For a given problem and one or more models, compute per-run totals of:
  total_cost, input_cost, output_cost, input_tokens, output_tokens,
  input_rate ($/1M), output_rate ($/1M)

then report mean ± std across runs, plus a most→least expensive ranking.

Saves:
  proc/<problem_name>/cost_stats.txt   (human-readable)
  proc/<problem_name>/cost_stats.json  (machine-readable)

Usage:
    # Single model
    python evals/model_cost_stats.py --problem_name tsp_constructive \\
        --model_name gemini-2.5-flash-lite --runs "1 2 3"

    # Multiple models (one --runs string per model, in order)
    python evals/model_cost_stats.py --problem_name tsp_constructive \\
        --models gemini-2.5-flash-lite gemini-2.5-flash multi_llm_ucb \\
        --runs "1 2 3" "1 2 3" "1"

    # Override results root
    python evals/model_cost_stats.py --problem_name tsp_constructive \\
        --models gemini-2.5-flash --runs "1 2" --results_dir my_results
"""

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

_KEY_TO_CANONICAL = {
    "gemini_flash_lite": "gemini-2.5-flash-lite",
    "gemini_flash":      "gemini-2.5-flash",
    "gemma_3_12b":       "gemma-3-12b-it",
    "gemma_3_27b":       "gemma-3-27b-it",
}

# Folder-name aliases: when looking up files for a model, also try these
# alternate folders. First existing wins per file.
MODEL_ALIASES = {
    "random":           ["multi_llm_random"],
    "multi_llm_random": ["random"],
}

# Canonical display name (also used as the stats dict key for merged folders).
MODEL_DISPLAY_NAMES = {
    "multi_llm_random": "random",
}


def _model_dirs(model_name: str) -> list:
    """Candidate folder names for a model: primary first, then aliases."""
    return [model_name] + MODEL_ALIASES.get(model_name, [])


def _canonical(model_id: str) -> str:
    s = str(model_id).strip()
    return _KEY_TO_CANONICAL.get(s, s)


def _get_rates(model_id: str, input_tokens: int = 0):
    """Return (in_rate, out_rate) in USD per 1 M tokens."""
    m = _canonical(model_id).lower()
    if "gemma" in m:
        return 0.0, 0.0
    if "flash-lite" in m:
        return 0.10, 0.40
    if "flash" in m:
        return 0.30, 2.50
    if "pro" in m:
        tier = "small" if input_tokens <= 200_000 else "large"
        return (1.25, 10.00) if tier == "small" else (2.50, 15.00)
    return 0.0, 0.0


def _call_metrics(entry: dict, dir_model: str):
    """Return per-call cost/token metrics, or None for placeholder entries."""
    in_tok  = entry.get("input_prompt_length", 0) or 0
    out_tok = entry.get("output_tokens", 0) or 0
    if in_tok == 0 and out_tok == 0:
        return None

    raw_model = entry.get("gen_model") or dir_model
    in_rate, out_rate = _get_rates(raw_model, in_tok)
    in_cost  = in_tok  * in_rate  / 1_000_000
    out_cost = out_tok * out_rate / 1_000_000

    return {
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "input_cost":    in_cost,
        "output_cost":   out_cost,
        "total_cost":    in_cost + out_cost,
        "input_rate":    in_rate,
        "output_rate":   out_rate,
    }


# ---------------------------------------------------------------------------
# Run discovery (mirrors build_table2.detect_runs_for)
# ---------------------------------------------------------------------------

def detect_runs_for(problem: str, model_name: str, inference_root: Path) -> list:
    """Sorted run numbers found under inference_results/<problem>/<model>/
    (and alias folders) as test_best_metric_run{N}.pkl — same source of truth
    used by build_table2.py.  Per-(problem, model)."""
    pattern = re.compile(r"^test_best_metric_run(\d+)\.pkl$")
    runs = set()
    for cand in _model_dirs(model_name):
        d = inference_root / problem / cand
        if not d.is_dir():
            continue
        for f in d.iterdir():
            m = pattern.match(f.name)
            if m:
                runs.add(int(m.group(1)))
    return sorted(runs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(path: Path, dir_model: str):
    """Aggregate all calls in one pkl file into a single per-run metrics dict."""
    with open(path, "rb") as f:
        log = pickle.load(f)

    agg = defaultdict(float)
    n_calls = 0
    rate_in_sum, rate_out_sum = 0.0, 0.0

    for entry in log:
        m = _call_metrics(entry, dir_model)
        if m is None:
            continue
        n_calls += 1
        for k in ("input_tokens", "output_tokens", "input_cost", "output_cost", "total_cost"):
            agg[k] += m[k]
        rate_in_sum  += m["input_rate"]
        rate_out_sum += m["output_rate"]

    if n_calls == 0:
        return None

    agg["input_rate"]  = rate_in_sum  / n_calls   # call-count-weighted average
    agg["output_rate"] = rate_out_sum / n_calls
    agg["n_calls"]     = n_calls
    return dict(agg)


def collect_stats(results_dir: Path, problem_name: str, model_runs: dict) -> dict:
    """
    Load runs for each model and compute mean ± std across runs.

    Args:
        results_dir : root results directory
        problem_name: e.g. "tsp_constructive"
        model_runs  : {model_name: [run_number, ...]}

    Returns:
        stats[model] = {metric: {"mean": float, "std": float}, "n_runs": int}
    """
    stats = {}
    METRICS = ("total_cost", "input_cost", "output_cost",
               "input_tokens", "output_tokens", "input_rate", "output_rate")

    for model, run_numbers in model_runs.items():
        run_records = []

        for run_num in run_numbers:
            path = None
            for cand in _model_dirs(model):
                p = results_dir / problem_name / cand / f"code_dict_run{run_num}.pkl"
                if p.exists():
                    path = p
                    break
            if path is None:
                miss = results_dir / problem_name / model / f"code_dict_run{run_num}.pkl"
                print(f"[!] Warning: {miss} not found — skipping run {run_num}.")
                continue
            rec = load_run(path, model)
            if rec is not None:
                run_records.append(rec)
                print(f"[+] Loaded {model} run {run_num}  ({rec['n_calls']} calls)")
            else:
                print(f"[!] {model} run {run_num} had no valid calls — skipping.")

        if not run_records:
            print(f"[!] No valid data for model '{model}'. Skipping.")
            continue

        model_stats = {"n_runs": len(run_records)}
        for metric in METRICS:
            vals = [r[metric] for r in run_records if metric in r]
            model_stats[metric] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)) if len(vals) > 1 else 0.0,
            }
        stats[model] = model_stats

    return stats


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "total_cost":    ("Total cost",    "USD"),
    "input_cost":    ("Input cost",    "USD"),
    "output_cost":   ("Output cost",   "USD"),
    "input_tokens":  ("Input tokens",  "tokens"),
    "output_tokens": ("Output tokens", "tokens"),
    "input_rate":    ("Input rate",    "$/1M tok"),
    "output_rate":   ("Output rate",   "$/1M tok"),
}


def _fmt(metric: str, val: float) -> str:
    if "cost" in metric:
        return f"${val:.6f}"
    if "rate" in metric:
        return f"{val:.4f}"
    return f"{val:,.1f}"


def write_text(problem_name: str, stats: dict, out_path: Path):
    lines = ["Model Cost Statistics", "=" * 72,
             f"Problem: {problem_name}", "=" * 72, ""]

    for model, mstats in stats.items():
        n_runs = mstats["n_runs"]
        lines.append(f"  Model: {model}  ({n_runs} run{'s' if n_runs != 1 else ''})")
        for metric, (label, unit) in METRIC_LABELS.items():
            if metric not in mstats:
                continue
            mean = mstats[metric]["mean"]
            std  = mstats[metric]["std"]
            lines.append(
                f"    {(label + ' (' + unit + ')').ljust(26)}"
                f"  {_fmt(metric, mean):>14}  ±  {_fmt(metric, std)}"
            )
        lines.append("")

    # ── Ranking ───────────────────────────────────────────────────────────────
    lines += ["=" * 72,
              "Ranking: Most → Least Expensive  (mean total cost per run, USD)",
              "=" * 72, ""]

    rows = []
    for model, mstats in stats.items():
        mean_cost = mstats.get("total_cost", {}).get("mean", 0.0)
        std_cost  = mstats.get("total_cost", {}).get("std",  0.0)
        n_runs    = mstats["n_runs"]
        rows.append((mean_cost, std_cost, model, n_runs))

    rows.sort(key=lambda r: r[0], reverse=True)
    rank_w = len(str(len(rows)))
    for rank, (mean_cost, std_cost, model, n_runs) in enumerate(rows, 1):
        lines.append(
            f"  {str(rank).rjust(rank_w)}.  ${mean_cost:.6f} ± ${std_cost:.6f}"
            f"   {model}  ({n_runs} run{'s' if n_runs != 1 else ''})"
        )

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[+] Saved text report → {out_path}")


def write_json(problem_name: str, stats: dict, out_path: Path):
    payload = {"problem": problem_name, "models": stats}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[+] Saved JSON report  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-model cost statistics for a given problem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--problem_name", type=str, required=True,
                        help="Problem directory name (e.g. tsp_constructive).")
    parser.add_argument("--results_dir", default="results",
                        help="Root results directory (default: results).")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory. Defaults to proc/<problem_name>/.")

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_name", type=str,
                             help="Single model name.")
    model_group.add_argument("--models", type=str, nargs="+",
                             help="One or more model names.")
    model_group.add_argument("--all", action="store_true",
                             help="Auto-discover all model subdirectories and runs "
                                  "under results/<problem_name>/.")

    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        help="Space-separated run numbers, one string per model. "
                             "Single model: --runs \"1 2 3\". "
                             "Multiple models: --runs \"1 2 3\" \"1 2\" \"1\". "
                             "If omitted, runs are auto-detected per (problem, model) "
                             "from inference_results/<problem>/<model>/test_best_metric_run*.pkl "
                             "(same source as build_table2.py --auto_runs).")

    args = parser.parse_args()

    root        = Path(__file__).resolve().parent.parent
    results_dir = root / args.results_dir
    out_dir     = root / (args.out_dir or f"proc/{args.problem_name}")

    if not results_dir.exists():
        print(f"[!] Results directory not found: {results_dir}")
        return

    inference_root = root / "inference_results"

    if args.all:
        # Discover models AND runs from inference_results so cost stats match
        # the (problem, model, run) triples that appear in the evaluation table.
        inference_problem_dir = inference_root / args.problem_name
        if not inference_problem_dir.exists():
            print(f"[!] Inference results directory not found: {inference_problem_dir}")
            return
        model_runs = {}
        for model_dir in sorted(inference_problem_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            key = MODEL_DISPLAY_NAMES.get(model_dir.name, model_dir.name)
            runs = detect_runs_for(args.problem_name, key, inference_root)
            if not runs:
                continue
            model_runs[key] = runs
        if not model_runs:
            print(f"[!] No inference runs found under {inference_problem_dir}")
            return
    else:
        models_to_process = [args.model_name] if args.model_name else args.models

        if args.runs is None:
            # Auto-detect runs per (problem, model) — same source of truth as
            # build_table2.py --auto_runs.  Sorted ascending → positional slots.
            model_runs = {}
            for model in models_to_process:
                key = MODEL_DISPLAY_NAMES.get(model, model)
                runs = detect_runs_for(args.problem_name, key, inference_root)
                if not runs:
                    print(f"[!] No runs found for {args.problem_name} / {key} "
                          f"under {inference_root / args.problem_name}; skipping.")
                    continue
                model_runs[key] = sorted(set(model_runs.get(key, [])).union(runs))
            if not model_runs:
                print("[-] Auto-detect found no runs. Exiting.")
                return
        else:
            if len(args.runs) != len(models_to_process):
                parser.error(
                    f"Number of --runs strings ({len(args.runs)}) must match "
                    f"number of models ({len(models_to_process)})."
                )
            model_runs = {}
            for model, runs_str in zip(models_to_process, args.runs):
                try:
                    runs_list = [int(r) for r in runs_str.split()]
                except ValueError:
                    parser.error(f"Invalid run numbers for '{model}': '{runs_str}'.")
                key = MODEL_DISPLAY_NAMES.get(model, model)
                model_runs[key] = sorted(set(model_runs.get(key, [])).union(runs_list))

    print(f"\n[*] Problem : {args.problem_name}")
    print(f"[*] Models  : {', '.join(model_runs.keys())}")
    print(f"[*] Results : {results_dir}")
    print(f"\nRuns read per (problem, model)  [slot N = positional alignment]:")
    for m, runs in model_runs.items():
        slots = "  ".join(f"slot{i}=run{r}" for i, r in enumerate(runs))
        print(f"  {m:30s}  {slots}")
    print()

    stats = collect_stats(results_dir, args.problem_name, model_runs)

    if not stats:
        print("[-] No data collected. Exiting.")
        return

    write_text(args.problem_name, stats, out_dir / "cost_stats.txt")
    write_json(args.problem_name, stats, out_dir / "cost_stats.json")


if __name__ == "__main__":
    main()
