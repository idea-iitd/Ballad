"""
eval_val.py  —  Evaluate each generated program on the validation set.

For each model/run, processes all calls in generation order and records:
  - call_id, prog_id
  - val instance-wise scores for this call's program
  - best-till-ith-call avg score (running maximum)
  - best_prog_id and its val scores / avg at that point

Uses a persistent disk cache keyed by func_id to avoid re-evaluating
identical programs across re-runs and across models.

Output: inference_results/{problem_name}/{model}/val_eval_run{N}.pkl
        (list of per-call dicts, ordered by call_id)

Usage:
    python inference/eval_val.py --problem_name tsp_constructive \\
        --models gemini-2.5-flash-lite --runs "1 2 3"
    python inference/eval_val.py --problem_name tsp_constructive \\
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

from evals.test_eval import Tester, load_run_logs

INFERENCE_DIR = PROJECT_ROOT / "inference_results"


def _import_helper(problem_name: str):
    return importlib.import_module(f"problems.{problem_name}.helpers")


def write_priority_func_val(problem_name: str, model_name: str, func_code: str) -> Path:
    priority_file = PROJECT_ROOT / "problems" / problem_name / f"priority_{model_name}_val_eval.py"
    with open(priority_file, "w") as f:
        f.write("import numpy as np\n")
        f.write("from copy import copy\n")
        f.write("from scipy.spatial import distance_matrix\n")
        f.write(func_code)
    return priority_file


def load_val_dataset(problem_name: str, problem_size, capacities, dataset_dir: Path):
    helper = _import_helper(problem_name)
    if problem_name == "ged":
        dataset_dict = helper.read_dataset(dataset_dir, dataset_name=str(problem_size), mode="val")
    elif problem_name == "bpp_online":
        dataset_dict = helper.read_dataset(
            dataset_dir, problem_size=str(problem_size), capacity=capacities, mode="val"
        )
    else:
        dataset_dict = helper.read_dataset(dataset_dir, problem_size=problem_size, mode="val")
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


def evaluate_on_val(func_code, func_id, problem_name, model_name, helper, dataset_dict,
                    persistent_cache, use_parallel, timeout_seconds, use_gpu_batch=False):
    """Evaluate a single program on val. Returns (scores, avg_score, eval_times, errors, timeouts)."""
    if func_id in persistent_cache:
        cached = persistent_cache[func_id]
        print(f"  [♻] Cache hit for func_id={func_id[:8]}...")
        return (cached['scores'], cached['avg_score'],
                cached['eval_times'], cached['errors'], cached['timeouts'])

    priority_file = write_priority_func_val(problem_name, model_name, func_code)
    tester = Tester(dataset_dict, helper, priority_file, timeout_seconds=timeout_seconds)
    torch.manual_seed(0)
    np.random.seed(0)
    if use_gpu_batch and hasattr(helper, 'evaluate_batch'):
        scores, avg_score, _, eval_times, errors, timeouts = tester.analyse_batch()
    else:
        scores, avg_score, _, eval_times, errors, timeouts = tester.analyse(use_parallel=use_parallel)

    persistent_cache[func_id] = {
        'scores': scores,
        'avg_score': avg_score,
        'eval_times': eval_times,
        'errors': errors,
        'timeouts': timeouts,
    }
    return scores, avg_score, eval_times, errors, timeouts


def process_run(log_data, problem_name, model_name, helper, dataset_dict,
                persistent_cache, use_parallel, timeout_seconds, use_gpu_batch=False):
    """
    Process all calls in a run log and return per-call records with best-till-i tracking.
    Records are ordered by call_id.
    """
    # Parse entries.  Call 0 (idx=0) is a dummy entry: gen_program_code/id are empty strings
    # because nothing was generated yet.  The actual initial seed program lives in best_gen_code/id
    # on that same entry.  For all other calls, gen_program_code/id hold the LLM-generated program.
    # Programs that errored during training (avg gen_scores = -inf) are not worth evaluating on val.
    entries_by_call = {}
    for idx, entry in enumerate(log_data):
        call_id = entry["call_id"]

        if idx == 0:  # seed: lives in best_gen_* fields
            func_code  = entry.get("best_gen_code")
            func_id    = entry.get("best_gen_id")
            gen_scores = entry.get("best_gen_scores", [])
        else:
            func_code  = entry.get("gen_program_code")
            func_id    = entry.get("gen_program_id")
            gen_scores = entry.get("gen_scores", [])

        if not func_id:
            continue  # nothing useful at all

        # Skip programs that errored on the training set (only applies to idx > 0)
        if idx > 0:
            gen_avg = np.mean(gen_scores) if gen_scores else -np.inf
            if not np.isfinite(gen_avg):
                func_code = None  # mark unevaluatable; still record the call

        entries_by_call[call_id] = {
            'func_code': func_code if func_code else None,
            'func_id': func_id,
        }

    sorted_call_ids = sorted(entries_by_call.keys())
    print(f"[+] Total calls in run: {len(sorted_call_ids)}")

    records = []
    best_val_avg = -np.inf
    best_prog_id = None
    best_val_scores = None
    best_prog_code = None

    for i, call_id in enumerate(sorted_call_ids):
        entry = entries_by_call[call_id]
        func_id = entry['func_id']
        func_code = entry['func_code']

        print(f"\n[{i+1}/{len(sorted_call_ids)}] call_id={call_id} func_id={func_id[:8]}...")

        if func_code is None:
            print(f"  [-] No func_code for func_id={func_id[:8]}..., skipping evaluation.")
            val_scores = None
            val_avg = None
        else:
            # Call 0 (seed program) is evaluated without GPU batch and without parallelism
            # to avoid stochastic ACO failures on the first call.
            is_call0 = (i == 0)
            scores, avg_score, _, _, _ = evaluate_on_val(
                func_code, func_id, problem_name, model_name, helper, dataset_dict,
                persistent_cache,
                use_parallel=False if is_call0 else use_parallel,
                timeout_seconds=timeout_seconds,
                use_gpu_batch=False if is_call0 else use_gpu_batch,
            )
            val_scores = scores
            val_avg = avg_score if np.isfinite(avg_score) else None

            # Higher raw avg_score is always better (helpers negate cost for min problems)
            if np.isfinite(avg_score) and avg_score > best_val_avg:
                best_val_avg = avg_score
                best_prog_id = func_id
                best_val_scores = scores
                best_prog_code = func_code

        record = {
            'call_no': i + 1,           # 1-indexed sequential position in sorted call order
            'call_id': call_id,
            'prog_id': func_id,
            'val_scores': val_scores,
            'val_avg_score': val_avg,
            'best_till_i': best_val_avg if np.isfinite(best_val_avg) else None,
            'best_prog_id': best_prog_id,
            'best_val_scores': best_val_scores,
            'best_val_avg': best_val_avg if np.isfinite(best_val_avg) else None,
            'best_prog_code': best_prog_code,
        }
        records.append(record)

    return records


def patch_run_call0(existing_records, log_data, problem_name, model_name, helper, dataset_dict,
                    persistent_cache, timeout_seconds):
    """
    Retroactively evaluate call 0 (the first call_id) on val if its val_scores are missing,
    then re-compute best_till tracking for all records without re-evaluating calls 1+.
    """
    records_by_call = {r['call_id']: r for r in existing_records}
    sorted_call_ids = sorted(records_by_call.keys())
    if not sorted_call_ids:
        return existing_records

    # Build log_by_call with position-based call 0 handling (idx=0 uses best_gen_*, others gen_program_*)
    log_by_call = {}
    for idx, e in enumerate(log_data):
        cid       = e['call_id']
        func_code = e.get('best_gen_code')    if idx == 0 else e.get('gen_program_code')
        func_id   = e.get('best_gen_id')      if idx == 0 else e.get('gen_program_id')
        if func_id:
            log_by_call[cid] = {'func_code': func_code, 'func_id': func_id}

    first_cid = sorted_call_ids[0]
    first_rec = records_by_call[first_cid]

    if first_rec['val_scores'] is None and first_cid in log_by_call:
        log_entry = log_by_call[first_cid]
        func_code = log_entry['func_code']
        func_id   = log_entry['func_id']
        if func_code and func_id:
            first_rec['prog_id'] = func_id  # fix empty-string prog_id from old pkl
            # Evict a stale -inf cache entry so the seed is re-evaluated with the fixed seed.
            # -inf on a hand-crafted seed is a stochastic ACO failure, not a genuine result.
            cached_entry = persistent_cache.get(func_id)
            if cached_entry is not None and not np.isfinite(cached_entry['avg_score']):
                print(f"  [↺] Evicting stale -inf val cache for func_id={func_id[:8]}...")
                del persistent_cache[func_id]
            print(f"[+] Patching call_id={first_cid}: evaluating on val (no GPU, no parallel)...")
            scores, avg_score, _, _, _ = evaluate_on_val(
                func_code, func_id, problem_name, model_name, helper, dataset_dict,
                persistent_cache, use_parallel=False, timeout_seconds=timeout_seconds,
                use_gpu_batch=False,
            )
            first_rec['val_scores'] = scores
            first_rec['val_avg_score'] = avg_score if np.isfinite(avg_score) else None
        else:
            print(f"[!] call_id={first_cid} has no func_code in log; skipping evaluation.")
    else:
        print(f"[=] call_id={first_cid} already has val scores or is absent from log. "
              "Re-computing best tracking only.")

    # Re-compute best_till tracking for all records using existing (and newly added) val scores
    best_val_avg = -np.inf
    best_prog_id = None
    best_val_scores = None
    best_prog_code = None

    updated = []
    for i, call_id in enumerate(sorted_call_ids):
        rec = records_by_call[call_id]
        if 'call_no' not in rec:
            rec['call_no'] = i + 1
        val_avg = rec['val_avg_score']

        if val_avg is not None and np.isfinite(val_avg) and val_avg > best_val_avg:
            best_val_avg = val_avg
            best_prog_id = rec['prog_id']
            best_val_scores = rec['val_scores']
            best_prog_code = log_by_call.get(call_id, {}).get('func_code')

        rec['best_till_i']     = best_val_avg if np.isfinite(best_val_avg) else None
        rec['best_prog_id']    = best_prog_id
        rec['best_val_scores'] = best_val_scores
        rec['best_val_avg']    = best_val_avg if np.isfinite(best_val_avg) else None
        rec['best_prog_code']  = best_prog_code
        updated.append(rec)

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all generated programs on the validation set per call."
    )
    parser.add_argument("--problem_name", required=True,
                        help="e.g. tsp_constructive, cvrp_aco, bpp_offline_aco")
    parser.add_argument("--models", nargs='+', required=False, default=None,
                        help="Model names to evaluate. Omit when using --patch_all.")
    parser.add_argument("--runs", nargs='+', required=False, default=None,
                        help="Run numbers per model as space-separated strings, one per model. "
                             "E.g. --runs \"1 2 3\" \"1 2\". Omit when using --patch_all.")
    parser.add_argument("--patch_all", action="store_true", default=False,
                        help="Auto-discover all models/runs from inference_results and patch "
                             "every existing val_eval_run pkl (implies --patch_existing).")
    parser.add_argument("--use_parallel", action="store_true", default=False,
                        help="Use multiprocessing for evaluation.")
    parser.add_argument("--use_gpu_batch", action="store_true", default=False,
                        help="Batch all instances into a single GPU ACO pass (CVRP only, fastest).")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-instance timeout in seconds (default: 60).")
    parser.add_argument("--patch_existing", action="store_true", default=False,
                        help="Patch existing val_eval_run pkl files: evaluate call 0 if its "
                             "val_scores are missing, then re-compute best_till tracking for "
                             "all subsequent records without re-evaluating them.")
    args = parser.parse_args()

    if args.patch_all:
        # Auto-discover every model dir that has at least one val_eval_run*.pkl
        import re as _re
        problem_inf_dir = INFERENCE_DIR / args.problem_name
        _run_pat = _re.compile(r"val_eval_run(\d+)\.pkl$")
        model_runs = {}
        if problem_inf_dir.exists():
            for model_dir in sorted(problem_inf_dir.iterdir()):
                if not model_dir.is_dir() or model_dir.name == "plots":
                    continue
                runs = sorted(
                    int(_run_pat.match(f.name).group(1))
                    for f in model_dir.iterdir()
                    if _run_pat.match(f.name)
                )
                if runs:
                    model_runs[model_dir.name] = runs
        if not model_runs:
            parser.error(
                f"--patch_all: no val_eval_run*.pkl files found under {problem_inf_dir}"
            )
        args.models = list(model_runs.keys())
        args.patch_existing = True
        print(f"[patch_all] Discovered: { {m: model_runs[m] for m in args.models} }")
    else:
        if not args.models or not args.runs:
            parser.error("--models and --runs are required (or use --patch_all)")
        if len(args.runs) != len(args.models):
            parser.error(
                f"--runs count ({len(args.runs)}) must match --models count ({len(args.models)})"
            )
        model_runs = {model: [int(r) for r in args.runs[i].split()]
                      for i, model in enumerate(args.models)}

    # Load problem size and capacity from training config
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
    if raw_capacity is not None:
        capacities = [int(c) for c in (raw_capacity if isinstance(raw_capacity, list) else [raw_capacity])]
    else:
        capacities = None

    # Cache key (same convention as select_val_top.py's make_config_key)
    if args.problem_name == "ged":
        config_key = str(problem_size)
    elif capacities:
        cap_str = "_".join(str(c) for c in sorted(capacities))
        config_key = f"{problem_size}_cap{cap_str}"
    else:
        config_key = str(problem_size)

    print(f"\n{'='*60}")
    print(f"  eval_val.py")
    print(f"  problem    : {args.problem_name}")
    print(f"  size       : {problem_size}")
    if capacities:
        print(f"  capacities : {capacities}")
    print(f"  config_key : {config_key}")
    print(f"{'='*60}\n")

    # Load val dataset
    dataset_dir = PROJECT_ROOT / "problems" / args.problem_name / "dataset"
    helper, dataset_dict = load_val_dataset(
        args.problem_name, problem_size, capacities, dataset_dir
    )
    print(f"[+] Loaded {dataset_dict['n_instances']} val instances")

    results_root = PROJECT_ROOT / "results"

    for model_name in args.models:
        print(f"\n[+] ===== Model: {model_name} =====")

        # Per-model cache avoids write races when multiple models run in parallel.
        cache_path = INFERENCE_DIR / args.problem_name / f"val_cache_{config_key}_{model_name}.pkl"
        persistent_cache = load_persistent_cache(cache_path)
        print(f"[+] Val cache ({model_name}): {len(persistent_cache)} entries from {cache_path}")

        model_results_dir = results_root / args.problem_name / model_name
        logs, loaded_runs = load_run_logs(model_results_dir, run_numbers=model_runs[model_name])

        out_dir = INFERENCE_DIR / args.problem_name / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for run_num, log_data in zip(loaded_runs, logs):
            print(f"\n[+] --- Run {run_num} ---")
            out_path = out_dir / f"val_eval_run{run_num}.pkl"

            if args.patch_existing:
                if not out_path.exists():
                    print(f"[!] No existing results at {out_path}. "
                          "Run without --patch_existing first. Skipping.")
                    continue
                with open(out_path, "rb") as f:
                    existing_records = pickle.load(f)
                print(f"[+] Loaded {len(existing_records)} existing records for patching.")
                records = patch_run_call0(
                    existing_records, log_data, args.problem_name, model_name, helper,
                    dataset_dict, persistent_cache, args.timeout,
                )
            else:
                records = process_run(
                    log_data, args.problem_name, model_name, helper, dataset_dict,
                    persistent_cache, args.use_parallel, args.timeout,
                    use_gpu_batch=args.use_gpu_batch,
                )

            records.sort(key=lambda r: r['call_no'])
            with open(out_path, "wb") as f:
                pickle.dump(records, f)
            print(f"[✓] Saved {len(records)} records → {out_path.relative_to(PROJECT_ROOT)}")

            # Save cache incrementally so progress is not lost on interruption
            save_persistent_cache(persistent_cache, cache_path)
            print(f"[✓] Cache saved: {len(persistent_cache)} entries")

    save_persistent_cache(persistent_cache, cache_path)
    print(f"\n[✓] All done. Final cache: {len(persistent_cache)} entries at {cache_path}")


if __name__ == "__main__":
    main()
