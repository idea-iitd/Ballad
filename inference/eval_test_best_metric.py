"""
eval_test_best_metric.py  —  Evaluate the single best val program on the test set.

For each model/run, reads val_eval_run{N}.pkl (from eval_val.py), takes the
best_prog_id from the FINAL record (= the program with the highest validation
score across all calls), and evaluates it once on the test set.

Shares the persistent test cache with eval_test_best.py so any program already
evaluated there is served instantly as a cache hit.

Output: inference_results/{problem_name}/{model}/test_best_metric_run{N}.pkl
        dict with keys:
            best_prog_id   : str
            best_val_avg   : float       (internal, higher = better)
            test_scores    : list[float] (per-instance internal scores)
            test_avg_score : float|None  (internal avg, higher = better)
            display_score  : float|None  (actual metric, lower = better)
            cached         : bool

After processing all runs per model, prints mean ± std of display_score.

Usage:
    python inference/eval_test_best_metric.py --problem_name tsp_constructive \\
        --models gemini-2.5-flash-lite --runs "1 2 3"
    python inference/eval_test_best_metric.py --problem_name tsp_constructive \\
        --models gemini-2.5-flash-lite gemini-2.5-flash \\
        --runs "1 2 3" "1 2" --use_parallel
"""

import sys
import pickle
import importlib
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.test_eval import Tester

INFERENCE_DIR = PROJECT_ROOT / "inference_results"

# Problems where helpers return −cost; negate back to get the actual metric.
NEGATE_PROBLEMS = {"cvrp_aco", "bpp_offline_aco", "tsp_constructive", "bpp_online", "ged"}


def _import_helper(problem_name: str):
    return importlib.import_module(f"problems.{problem_name}.helpers")


def write_priority_func(problem_name: str, model_name: str, func_code: str) -> Path:
    priority_file = (
        PROJECT_ROOT / "problems" / problem_name / f"priority_{model_name}_test_metric.py"
    )
    with open(priority_file, "w") as f:
        f.write("import numpy as np\n")
        f.write("from copy import copy\n")
        f.write("from scipy.spatial import distance_matrix\n")
        f.write(func_code)
    return priority_file


def load_test_dataset(problem_name: str, problem_size, capacities, dataset_dir: Path):
    helper = _import_helper(problem_name)
    if problem_name == "ged":
        dataset_dict = helper.read_dataset(
            dataset_dir, dataset_name=str(problem_size), mode="test"
        )
    elif problem_name == "bpp_online":
        dataset_dict = helper.read_dataset(
            dataset_dir, problem_size=str(problem_size), capacity=capacities, mode="test"
        )
    else:
        dataset_dict = helper.read_dataset(
            dataset_dir, problem_size=problem_size, mode="test"
        )
    return helper, dataset_dict


def load_persistent_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_persistent_cache(cache: dict, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def evaluate_on_test(
    func_code, func_id, problem_name, model_name, helper, dataset_dict,
    persistent_cache, use_parallel, timeout_seconds, use_gpu_batch=False,
):
    """Evaluate one program on the test set. Returns 7-tuple including cached flag."""
    if func_id in persistent_cache:
        cached = persistent_cache[func_id]
        print(f"  [♻] Cache hit for func_id={func_id[:8]}...")
        return (
            cached["scores"], cached["avg_score"], cached["eval_times"],
            cached["errors"], cached["timeouts"], cached.get("avg_eval_time"), True,
        )

    priority_file = write_priority_func(problem_name, model_name, func_code)
    tester = Tester(dataset_dict, helper, priority_file, timeout_seconds=timeout_seconds)
    torch.manual_seed(0)
    np.random.seed(0)
    if use_gpu_batch and hasattr(helper, "evaluate_batch"):
        scores, avg_score, _, eval_times, errors, timeouts = tester.analyse_batch()
    else:
        scores, avg_score, _, eval_times, errors, timeouts = tester.analyse(
            use_parallel=use_parallel
        )

    tracked = [t for t in eval_times if t is not None]
    avg_eval_time = float(np.mean(tracked)) if tracked else None

    persistent_cache[func_id] = {
        "scores": scores,
        "avg_score": avg_score,
        "eval_times": eval_times,
        "errors": errors,
        "timeouts": timeouts,
        "avg_eval_time": avg_eval_time,
    }
    return scores, avg_score, eval_times, errors, timeouts, avg_eval_time, False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the single best val program (final best_prog_id) on the test set."
    )
    parser.add_argument("--problem_name", required=True,
                        help="e.g. tsp_constructive, cvrp_aco, bpp_offline_aco")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to evaluate.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run numbers per model as space-separated strings, one per model. "
                             "E.g. --runs \"1 2 3\" \"1 2\"")
    parser.add_argument("--use_parallel", action="store_true", default=False,
                        help="Use multiprocessing for test evaluation.")
    parser.add_argument("--use_gpu_batch", action="store_true", default=False,
                        help="Batch GPU ACO pass (CVRP only).")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-instance timeout in seconds (default: 60).")
    args = parser.parse_args()

    if len(args.runs) != len(args.models):
        parser.error(
            f"--runs count ({len(args.runs)}) must match --models count ({len(args.models)})"
        )
    model_runs = {
        model: [int(r) for r in args.runs[i].split()]
        for i, model in enumerate(args.models)
    }

    # Load problem config
    problem_cfg_path = PROJECT_ROOT / "cfg" / "problem" / f"{args.problem_name}.yaml"
    if not problem_cfg_path.exists():
        parser.error(f"Problem config not found: {problem_cfg_path}")
    with open(problem_cfg_path) as f:
        problem_cfg = yaml.safe_load(f)

    problem_size = problem_cfg.get("problem_size", "")
    if not problem_size and args.problem_name == "ged":
        problem_size = problem_cfg.get("dataset_name", "")
    if not problem_size:
        parser.error(f"'problem_size' not set in {problem_cfg_path}")

    raw_capacity = problem_cfg.get("capacity", None)
    capacities = None
    if raw_capacity is not None:
        capacities = [
            int(c) for c in (raw_capacity if isinstance(raw_capacity, list) else [raw_capacity])
        ]

    if args.problem_name == "ged":
        config_key = str(problem_size)
    elif capacities:
        cap_str = "_".join(str(c) for c in sorted(capacities))
        config_key = f"{problem_size}_cap{cap_str}"
    else:
        config_key = str(problem_size)

    negate = args.problem_name in NEGATE_PROBLEMS

    print(f"\n{'='*60}")
    print(f"  eval_test_best_metric.py")
    print(f"  problem    : {args.problem_name}")
    print(f"  size       : {problem_size}")
    if capacities:
        print(f"  capacities : {capacities}")
    print(f"  config_key : {config_key}")
    print(f"  negate     : {negate}")
    print(f"{'='*60}\n")

    dataset_dir = PROJECT_ROOT / "problems" / args.problem_name / "dataset"
    helper, dataset_dict = load_test_dataset(
        args.problem_name, problem_size, capacities, dataset_dir
    )
    print(f"[+] Loaded {dataset_dict['n_instances']} test instances")

    for model_name in args.models:
        print(f"\n[+] ===== Model: {model_name} =====")
        out_dir = INFERENCE_DIR / args.problem_name / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-model cache avoids write races when multiple models run in parallel.
        cache_path = INFERENCE_DIR / args.problem_name / f"test_cache_{config_key}_{model_name}.pkl"
        persistent_cache = load_persistent_cache(cache_path)
        print(f"[+] Test cache ({model_name}): {len(persistent_cache)} entries from {cache_path}")

        run_display_scores = []

        for run_num in model_runs[model_name]:
            val_path = (
                INFERENCE_DIR / args.problem_name / model_name / f"val_eval_run{run_num}.pkl"
            )
            if not val_path.exists():
                print(
                    f"[!] Val eval not found: {val_path}. "
                    "Run eval_val.py first. Skipping run."
                )
                continue

            with open(val_path, "rb") as f:
                val_records = pickle.load(f)

            # The final record (sorted by call_no) holds the best prog across all calls.
            val_records_sorted = sorted(val_records, key=lambda r: r["call_no"])
            final_record = val_records_sorted[-1]

            best_prog_id   = final_record["best_prog_id"]
            best_prog_code = final_record["best_prog_code"]
            best_val_avg   = final_record["best_val_avg"]

            id_str = best_prog_id[:8] if best_prog_id else "None"
            print(
                f"\n[+] Run {run_num}: best_prog_id={id_str}...  best_val_avg={best_val_avg}"
            )

            if best_prog_id is None or best_prog_code is None:
                print(f"[!] No valid best program in run {run_num}. Skipping.")
                continue

            (test_scores, test_avg, eval_times,
             errors, timeouts, avg_eval_time, cached) = evaluate_on_test(
                best_prog_code, best_prog_id, args.problem_name, model_name,
                helper, dataset_dict, persistent_cache,
                args.use_parallel, args.timeout, use_gpu_batch=args.use_gpu_batch,
            )

            test_avg_finite = test_avg if (test_avg is not None and np.isfinite(test_avg)) else None
            display_score = (
                -test_avg_finite if (negate and test_avg_finite is not None) else test_avg_finite
            )

            result = {
                "best_prog_id":   best_prog_id,
                "best_val_avg":   best_val_avg,
                "test_scores":    test_scores,
                "test_avg_score": test_avg_finite,
                "display_score":  display_score,
                "cached":         cached,
            }

            out_path = out_dir / f"test_best_metric_run{run_num}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(result, f)

            status    = "✓" if display_score is not None else "✗"
            score_str = f"{display_score:.4f}" if display_score is not None else "N/A"
            print(f"  {status} display_score={score_str}  cached={cached}")
            print(f"[✓] Saved → {out_path.relative_to(PROJECT_ROOT)}")

            save_persistent_cache(persistent_cache, cache_path)
            if display_score is not None:
                run_display_scores.append(display_score)

        if run_display_scores:
            avg = np.mean(run_display_scores)
            std = np.std(run_display_scores)
            print(
                f"\n[=] {model_name}:  "
                f"display_score = {avg:.4f} ± {std:.4f}  "
                f"(over {len(run_display_scores)} runs)"
            )
        else:
            print(f"\n[=] {model_name}: no valid scores collected")

    save_persistent_cache(persistent_cache, cache_path)
    print(f"\n[✓] All done. Cache: {len(persistent_cache)} entries at {cache_path}")


if __name__ == "__main__":
    main()
