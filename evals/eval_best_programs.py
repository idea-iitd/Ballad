"""
eval_best_programs.py  —  Evaluate programs from best_programs/ on val and/or test sets.

Each file in best_programs/ is named {problem_name}_run{N}.py and contains a
single priority function.  This script loads the corresponding dataset, evaluates
the function, and reports the display score (actual metric, lower = better).

Usage
-----
# Evaluate all best programs on both splits
python evals/eval_best_programs.py --all --mode both

# Evaluate all best programs for one problem
python evals/eval_best_programs.py --problem tsp_constructive --mode test

# Evaluate specific files
python evals/eval_best_programs.py \
    --programs best_programs/tsp_constructive_run1.py best_programs/cvrp_aco_run2.py \
    --mode val

# Parallel evaluation with longer timeout
python evals/eval_best_programs.py --all --mode both --use_parallel --timeout 120

# GPU-batched evaluation (CVRP only)
python evals/eval_best_programs.py --problem cvrp_aco --mode test --use_gpu_batch
"""

import argparse
import importlib
import sys
import yaml
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.test_eval import Tester

# All problems negate internally (helpers return −cost; negate back for display)
NEGATE_PROBLEMS = {"cvrp_aco", "bpp_offline_aco", "tsp_constructive", "bpp_online", "ged"}

KNOWN_PROBLEMS = [
    "bpp_offline_aco",
    "bpp_online",
    "cvrp_aco",
    "ged",
    "tsp_constructive",
]


def parse_problem_from_filename(path: Path) -> tuple[str, str]:
    """Return (problem_name, label) parsed from a best_programs filename."""
    stem = path.stem  # e.g. tsp_constructive_run1
    for prob in KNOWN_PROBLEMS:
        if stem.startswith(prob):
            label = stem[len(prob):].lstrip("_") or stem
            return prob, label
    raise ValueError(
        f"Cannot determine problem name from '{path.name}'. "
        f"Expected prefix from: {KNOWN_PROBLEMS}"
    )


def load_problem_config(problem_name: str) -> dict:
    cfg_path = PROJECT_ROOT / "cfg" / "problem" / f"{problem_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Problem config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_dataset(problem_name: str, cfg: dict, mode: str) -> dict:
    helper = importlib.import_module(f"problems.{problem_name}.helpers")
    dataset_dir = PROJECT_ROOT / "problems" / problem_name / "dataset"

    raw_cap = cfg.get("capacity")
    capacities = (
        [int(c) for c in (raw_cap if isinstance(raw_cap, list) else [raw_cap])]
        if raw_cap is not None else None
    )

    if problem_name == "ged":
        dataset_name = str(cfg.get("dataset_name", cfg.get("problem_size", "")))
        dataset_dict = helper.read_dataset(dataset_dir, dataset_name=dataset_name, mode=mode)
    elif problem_name == "bpp_online":
        problem_size = str(cfg.get("problem_size", ""))
        dataset_dict = helper.read_dataset(
            dataset_dir, problem_size=problem_size, capacity=capacities, mode=mode
        )
    else:
        problem_size = cfg.get("problem_size", "")
        dataset_dict = helper.read_dataset(dataset_dir, problem_size=problem_size, mode=mode)

    return dataset_dict


def write_priority_file(problem_name: str, func_code: str) -> Path:
    priority_file = PROJECT_ROOT / "problems" / problem_name / "_priority_eval_best.py"
    with open(priority_file, "w") as f:
        f.write("import numpy as np\n")
        f.write("from copy import copy\n")
        f.write("from scipy.spatial import distance_matrix\n")
        f.write(func_code)
    return priority_file


def evaluate_program(
    problem_name: str,
    func_code: str,
    dataset_dict: dict,
    mode: str,
    use_parallel: bool,
    use_gpu_batch: bool,
    timeout_seconds: int,
) -> float | None:
    helper = importlib.import_module(f"problems.{problem_name}.helpers")
    priority_file = write_priority_file(problem_name, func_code)

    torch.manual_seed(0)
    np.random.seed(0)

    tester = Tester(dataset_dict, helper, priority_file, timeout_seconds=timeout_seconds)

    if use_gpu_batch and hasattr(helper, "evaluate_batch"):
        scores, avg_score, _, _, _, _ = tester.analyse_batch()
    else:
        scores, avg_score, _, _, _, _ = tester.analyse(use_parallel=use_parallel)

    priority_file.unlink(missing_ok=True)

    if avg_score is None or not np.isfinite(avg_score):
        return None

    negate = problem_name in NEGATE_PROBLEMS
    return -avg_score if negate else avg_score


def collect_files(args) -> list[Path]:
    if args.programs:
        return [Path(p) for p in args.programs]
    best_programs_dir = PROJECT_ROOT / "best_programs"
    if args.problem:
        files = sorted(best_programs_dir.glob(f"{args.problem}_*.py"))
        if not files:
            raise FileNotFoundError(
                f"No best_programs files found for problem '{args.problem}'"
            )
        return files
    # --all
    return sorted(best_programs_dir.glob("*.py"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best_programs/ on val and/or test splits."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--programs", nargs="+",
                     help="Specific best_programs .py files to evaluate.")
    src.add_argument("--problem",
                     help="Evaluate all best_programs files for this problem.")
    src.add_argument("--all", action="store_true",
                     help="Evaluate every file in best_programs/.")

    parser.add_argument("--mode", choices=["val", "test", "both"], default="both",
                        help="Which dataset split(s) to evaluate (default: both).")
    parser.add_argument("--use_parallel", action="store_true",
                        help="Parallelise evaluation across instances.")
    parser.add_argument("--use_gpu_batch", action="store_true",
                        help="GPU-batched evaluation (CVRP only).")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-instance timeout in seconds (default: 60).")
    args = parser.parse_args()

    modes = ["val", "test"] if args.mode == "both" else [args.mode]
    files = collect_files(args)

    if not files:
        print("[!] No files to evaluate.")
        return

    # Group results by problem for summary
    results: dict[str, dict[str, list]] = {}  # problem -> mode -> list of scores

    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  Evaluating {len(files)} program(s)  |  mode={args.mode}")
    print(SEP)

    for prog_file in files:
        prog_file = Path(prog_file)
        try:
            problem_name, label = parse_problem_from_filename(prog_file)
        except ValueError as e:
            print(f"[!] Skipping {prog_file.name}: {e}")
            continue

        print(f"\n[>] {prog_file.name}  ({problem_name})")

        cfg = load_problem_config(problem_name)
        func_code = prog_file.read_text()

        if problem_name not in results:
            results[problem_name] = {m: [] for m in modes}

        for mode in modes:
            print(f"    [{mode}] Loading dataset...")
            try:
                dataset_dict = load_dataset(problem_name, cfg, mode)
            except Exception as e:
                print(f"    [{mode}] Dataset load failed: {e}")
                continue

            n = dataset_dict.get("n_instances", "?")
            print(f"    [{mode}] {n} instances  |  evaluating...")

            score = evaluate_program(
                problem_name, func_code, dataset_dict, mode,
                args.use_parallel, args.use_gpu_batch, args.timeout,
            )

            if score is None:
                print(f"    [{mode}] result: FAILED (all instances errored or timed out)")
            else:
                print(f"    [{mode}] result: {score:.4f}")
                results[problem_name][mode].append(score)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY  (mean ± std over programs evaluated per problem)")
    print(SEP)
    print(f"{'Problem':<22}  {'Split':<6}  {'N':>3}  {'Mean':>10}  {'Std':>10}")
    print("-" * 60)
    for prob in KNOWN_PROBLEMS:
        if prob not in results:
            continue
        for mode in modes:
            scores = results[prob][mode]
            if not scores:
                print(f"{prob:<22}  {mode:<6}  {'0':>3}  {'N/A':>10}  {'N/A':>10}")
            else:
                mean = np.mean(scores)
                std  = np.std(scores)
                n    = len(scores)
                print(f"{prob:<22}  {mode:<6}  {n:>3}  {mean:>10.4f}  {std:>10.4f}")
    print(SEP)


if __name__ == "__main__":
    main()
