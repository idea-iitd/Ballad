import copy
import random
from tqdm import tqdm
from typing import Any, Tuple
from scipy.spatial import distance_matrix
from timeout_decorator import timeout
import importlib
import numpy as np
import argparse
from pathlib import Path
import time
import os
import pickle
from multiprocessing import cpu_count
import multiprocessing as mp
import pandas as pd
import yaml

class FunctionScoreCache:
  """
  Cache to avoid re-evaluating the same function code.
  Maps function_id -> per-instance and summary evaluation data.
  """
  def __init__(self):
    self.cache = {}

  def get(self, func_id):
    return self.cache.get(func_id)

  def set(self, func_id, scores, avg_score, eval_time, eval_times, errors, timeouts):
    self.cache[func_id] = {
        'scores': scores,
        'avg_score': avg_score,
        'eval_time': eval_time,
        'eval_times': eval_times,
        'errors': errors,
        'timeouts': timeouts,
    }

  def print_stats(self):
    print(f"[+] Function Score Cache Stats:")
    print(f"    Cached functions: {len(self.cache)}")
    for func_id, data in self.cache.items():
      avg_score = data['avg_score']
      status = "✓ Valid" if avg_score != -np.inf else "✗ Errored"
      print(f"    - func_id {func_id}: avg_score={avg_score:.2f} {status}")

class TestBed:
  """TestBed for executing generated code."""
  def __init__(self, helper, priority_file: Any):
    self.helper = helper
    self.priority_file = priority_file

  def run(self, test_input, timeout_seconds):
    if hasattr(test_input, "_fields"):  # NamedTuple check - for eg: BPPInstance
      code_to_execute = "result = evaluate(test_input, priority_file)"
    else:
      code_to_execute = "result = evaluate(*test_input, priority_file)"

    execution_succeeded = False
    result = None

    exec_globals = {
        "np": np,
        "copy": copy,
        "distance_matrix": distance_matrix,
    }
    exec_locals = {
        "priority": self.helper.priority,
        "solve": self.helper.solve,
        "evaluate": self.helper.evaluate,
        "result": result,
        "test_input": test_input,
        "priority_file": self.priority_file,
    }

    failure_reason = None

    @timeout(timeout_seconds, timeout_exception=TimeoutError)
    def _exec():
      exec(code_to_execute, exec_globals, exec_locals)

    try:
      _exec()
      result = exec_locals.get("result")  # (-cost, func_name, func_code)

      if result is not None and isinstance(result, tuple) and len(result) == 3:
        cost, func_name, func_code = result
      else:
        cost, func_name, func_code = np.inf, None, None

      print("Code executed successfully, cost= ", cost)
      execution_succeeded = True
    except TimeoutError:
      cost, func_name, func_code = np.inf, None, None
      failure_reason = 'timeout'
      print(f"Execution timed out ({timeout_seconds} seconds)")
      execution_succeeded = False
    except Exception as e:
      cost, func_name, func_code = np.inf, None, f"Error: {e}"
      failure_reason = 'error'
      execution_succeeded = False

    return cost, func_name, func_code, execution_succeeded, failure_reason

def _seed_for_instance(idx: int) -> None:
    """Seed all RNG sources deterministically for instance idx.
    Called in both parallel and sequential paths so results are identical."""
    random.seed(idx)
    np.random.seed(idx)
    try:
        import torch
        torch.manual_seed(idx)
    except ImportError:
        pass


def _run_single_test(args: Tuple[int, Any, str, Any, int]) -> Tuple[int, float, str, str, bool, float]:
    """Helper function to run a single test case. Returns (index, score, func_name, func_code, runs_ok, eval_time)."""
    idx, current_input, helper_module_name, priority_file, timeout_seconds = args

    _seed_for_instance(idx)

    helper = importlib.import_module(helper_module_name)
    testbed = TestBed(helper, priority_file)

    if not isinstance(current_input, tuple):
        current_input = (current_input,)

    print(f"\n----------------------------------------------")
    print(f"Input idx: {idx}")

    inst_start = time.time()
    test_output, func_name, func_code, runs_ok, failure_reason = testbed.run(current_input, timeout_seconds)
    eval_time = time.time() - inst_start

    if runs_ok and test_output is not None:
        if not isinstance(test_output, (int, float)):
            print(f"Error: function did not return int/float score for test {idx}")
            return (idx, -np.inf, func_name, func_code, False, 'error', eval_time)
        print(f"Func Name: {func_name}")
        print(f"Func Code: \n{func_code}")
        print(f"Score: {test_output}")
        return (idx, test_output, func_name, func_code, True, None, eval_time)
    else:
        print(f"Function execution failed for test case {idx}.")
        print(f"Func Name: {func_name}")
        print(f"Func Code: \n{func_code}")
        print("Score: -inf")
        return (idx, -np.inf, func_name, func_code, False, failure_reason, eval_time)
    
class Tester:
  """Class that analyses functions generated by LLMs."""

  def __init__(
      self,
      dataset_dict: dict,
      helper: Any,
      priority_file: Any,
      timeout_seconds: int = 60,
      num_workers: int = None,
  ):
    self._inputs = helper.get_inputs(dataset_dict)
    self._num_inputs = dataset_dict['n_instances']
    self._timeout_seconds = timeout_seconds
    self._testbed = TestBed(helper, priority_file)
    self._num_workers = num_workers or max(1, cpu_count() - 5)
    self._helper_module_name = helper.__name__
    self._priority_file = priority_file

  def analyse(self, use_parallel: bool = False):
    """Executes the priority function on all test inputs. Returns per-instance results."""

    scores_per_test = {}
    eval_times_per_test = {}
    errors_per_test = {}
    timeouts_per_test = {}

    if use_parallel and self._num_inputs > 1:
      test_args = [
          (i, self._inputs[i], self._helper_module_name, self._priority_file, self._timeout_seconds)
          for i in range(self._num_inputs)
      ]
      ctx = mp.get_context('fork')
      with ctx.Pool(processes=self._num_workers) as pool:
        results = pool.imap_unordered(_run_single_test, test_args)
        try:
          for idx, score, func_name, _, runs_ok, failure_reason, eval_time in tqdm(
              results, total=self._num_inputs, desc="Processing inputs"
          ):
            scores_per_test[idx] = score
            errors_per_test[idx] = (failure_reason == 'error')
            timeouts_per_test[idx] = (failure_reason == 'timeout')
            eval_times_per_test[idx] = eval_time
        except Exception as e:
          print(f"[!] Worker pool error: {e}. Partial results retained for {len(scores_per_test)} inputs.")
          pool.terminate()
      # Fill missing entries (from crashed workers) as errors
      for i in range(self._num_inputs):
        if i not in scores_per_test:
          scores_per_test[i] = -np.inf
          errors_per_test[i] = True
          timeouts_per_test[i] = False
          eval_times_per_test[i] = None
    else:
      for i in range(self._num_inputs):
        current_input = self._inputs[i]
        if not isinstance(current_input, tuple):
          current_input = (current_input,)

        print("\n----------------------------------------------")
        print("Input idx: ", i)
        _seed_for_instance(i)
        inst_start = time.time()
        test_output, func_name, _, runs_ok, failure_reason = self._testbed.run(
            current_input, self._timeout_seconds
        )
        eval_times_per_test[i] = time.time() - inst_start
        errors_per_test[i] = (failure_reason == 'error')
        timeouts_per_test[i] = (failure_reason == 'timeout')

        if runs_ok and test_output is not None:
          if not isinstance(test_output, (int, float)):
            raise ValueError('@function.run did not return an int/float score.')
          scores_per_test[i] = test_output
          print("Func Name: ", func_name)
          print("Score: ", test_output)
        else:
          print(f"Function execution failed for test case {i} (reason={failure_reason}).")
          scores_per_test[i] = -np.inf

    # Build ordered lists
    individual_scores = [scores_per_test[i] for i in range(self._num_inputs)]
    eval_times = [eval_times_per_test[i] for i in range(self._num_inputs)]
    errors = [errors_per_test[i] for i in range(self._num_inputs)]
    timeouts = [timeouts_per_test[i] for i in range(self._num_inputs)]

    tracked = [t for t in eval_times if t is not None]
    avg_eval_time = np.mean(tracked) if tracked else 0.0

    if individual_scores and all(np.isfinite(s) for s in individual_scores):
      curr_avg_score = np.mean(individual_scores)
      print("Current Avg Score: ", curr_avg_score)
    else:
      curr_avg_score = -np.inf

    return individual_scores, curr_avg_score, avg_eval_time, eval_times, errors, timeouts

  def analyse_batch(self):
    """
    GPU-batched evaluation for CVRP: all instances run in a single BatchedACO pass.

    Requires helper.evaluate_batch(node_positions_list, demand_list, priority_file).
    No subprocesses are spawned — designed for inference speed, not training isolation.

    Returns the same 6-tuple as analyse().
    """
    helper = importlib.import_module(self._helper_module_name)
    if not hasattr(helper, 'evaluate_batch'):
      raise RuntimeError(
          f"{self._helper_module_name} has no evaluate_batch(); "
          "use analyse() or add evaluate_batch to the helper."
      )

    t0 = time.time()
    try:
      # Each helper's evaluate_batch accepts (inputs, priority_file) where inputs
      # is whatever get_inputs() returns — e.g. list of (pos,dem) tuples for CVRP
      # or list of BPPInstance for BPP.
      neg_costs, _, _ = helper.evaluate_batch(
          self._inputs, self._priority_file
      )
      individual_scores = list(neg_costs.astype(float))
    except Exception as e:
      print(f"[!] evaluate_batch failed: {e}")
      individual_scores = [-np.inf] * self._num_inputs

    total_time = time.time() - t0
    avg_eval_time = total_time / max(self._num_inputs, 1)
    eval_times = [avg_eval_time] * self._num_inputs
    errors   = [not np.isfinite(s) for s in individual_scores]
    timeouts = [False] * self._num_inputs

    if all(np.isfinite(s) for s in individual_scores):
      curr_avg_score = float(np.mean(individual_scores))
      print(f"[GPU batch] {self._num_inputs} instances in {total_time:.2f}s  avg_score={curr_avg_score:.4f}")
    else:
      curr_avg_score = -np.inf
      print(f"[GPU batch] failed — {sum(errors)} / {self._num_inputs} instances errored")

    return individual_scores, curr_avg_score, avg_eval_time, eval_times, errors, timeouts

def write_priority_func(problem_name: str, model_name: str, priority_func: str):
  problem_dir = Path("problems") / problem_name
  priority_file = problem_dir / f"priority_{model_name}_test.py"

  # --- Write function to priority file ---
  with open(priority_file, "w") as f:
    f.write("import numpy as np\n")
    f.write("from copy import copy\n")
    f.write("from scipy.spatial import distance_matrix\n")
    f.write(priority_func)

  print(f"[+] Wrote {priority_file}")

  return priority_file

def _import_helper(problem_name: str):
  """
  Import helpers.py from problems/<problem_name> as a package.
  Requires problems/ and each problem folder to have __init__.py
  """
  module_name = f"problems.{problem_name}.helpers"
  return importlib.import_module(module_name)

def load_run_logs(results_dir, run_numbers=None):
  """
  Load specified run logs from a directory.
  
  Args:
      results_dir: Path to directory containing log files
      run_numbers: List of run numbers to load (e.g., [1, 2, 3]). 
                    If None, loads all available runs up to 3.
  
  Returns:
      List of log data, list of run numbers actually loaded
  """
  logs = []
  loaded_runs = []
  
  if run_numbers is None:
    # Auto-detect available runs (checks up to run 10)
    run_numbers = []
    for i in range(1, 4):
      file_path = results_dir / f"code_dict_run{i}.pkl"
      if file_path.exists():
        run_numbers.append(i)
  
  for run_num in run_numbers:
    file_path = results_dir / f"code_dict_run{run_num}.pkl"
    if file_path.exists():
      with open(file_path, "rb") as f:
        log_data = pickle.load(f)
        logs.append(log_data)
        loaded_runs.append(run_num)
        print(f"[+] Loaded run {run_num} with {len(log_data)} calls")
    else:
      print(f"[!] Warning: Run {run_num} not found at {file_path}")
  
  if not logs:
      print(f"[-] No log files found in {results_dir}")
  
  return logs, loaded_runs

def extract_training_data(logs, loaded_runs):
    """
    Extract all generated programs from logs (not just the running best).
    Returns: dict mapping run_idx -> {call_id: (func_code, func_id)}
    """
    all_programs = {}

    for run_idx, log in zip(loaded_runs, logs):
      all_programs[run_idx] = {}
      for idx, entry in enumerate(log):
        call_id = entry["call_id"]

        if idx == 0:  # dummy entry: seed lives in best_gen_*
          func_code  = entry.get("best_gen_code")
          func_id    = entry.get("best_gen_id")
          gen_scores = entry.get("best_gen_scores", [])
        else:
          func_code  = entry["gen_program_code"]
          gen_scores = entry.get("gen_scores", [])
          func_id    = entry["gen_program_id"]

        curr_avg_score = np.mean(gen_scores) if gen_scores else -np.inf

        if np.isfinite(curr_avg_score) and (func_code is not None):
          all_programs[run_idx][call_id] = (func_code, func_id)
        else:
          all_programs[run_idx][call_id] = (None, func_id)

    return all_programs


def create_excel_reports(problem_name, model_name, problem_size, loaded_runs, test_results_dir):
  """
  Create one Excel file per model. One sheet per run.
  Each sheet has one row per evaluated program with full per-instance detail.
  """
  excel_dir = test_results_dir / problem_name / model_name
  excel_dir.mkdir(parents=True, exist_ok=True)
  excel_path = excel_dir / f"test_results_{problem_size}.xlsx"

  any_written = False

  with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for run in loaded_runs:
      pkl_path = test_results_dir / problem_name / model_name / f"test_res_{run}.pkl"

      if not pkl_path.exists():
        print(f"[!] Warning: Results file not found: {pkl_path}")
        continue

      with open(pkl_path, "rb") as f:
        results_dict = pickle.load(f)

      if not results_dict:
        print(f"[!] Empty results for run {run}, skipping sheet.")
        continue

      rows = []
      for call_id in sorted(results_dict.keys()):
        d = results_dict[call_id]
        scores = d.get('scores', [])
        eval_times = d.get('eval_times', [])
        errors = d.get('errors', [])
        timeouts = d.get('timeouts', [])

        row = {
            'call_id': call_id,
            'func_id': d.get('func_id'),
            'func_code': d.get('func_code'),
            'avg_test_score': d.get('avg_score'),
        }

        for i, score in enumerate(scores):
          row[f'score_inst_{i}'] = score
        for i, et in enumerate(eval_times):
          row[f'eval_time_inst_{i}'] = et
        for i, err in enumerate(errors):
          row[f'error_inst_{i}'] = err
        for i, to in enumerate(timeouts):
          row[f'timeout_inst_{i}'] = to

        rows.append(row)

      df = pd.DataFrame(rows)
      sheet_name = f"run{run}"
      df.to_excel(writer, index=False, sheet_name=sheet_name)
      print(f"[+] Written sheet '{sheet_name}' with {len(rows)} rows")
      any_written = True

  if any_written:
    print(f"[+] Saved Excel report: {excel_path}")
  else:
    print(f"[!] No data written for {model_name}")


def main():
  """Main function to parse arguments and run evaluation."""
  parser = argparse.ArgumentParser(description="Evaluate test performance from experiment logs.")
  
  parser.add_argument(
      "--problem_name", type=str, required=True,
      help="Name of the problem (e.g., tsp_constructive, cvrp_aco, bpp_online, bpp_offline_aco)."
  )

  parser.add_argument(
      "--models", type=str, nargs='+', required=True,
      help="Name(s) of model(s) to evaluate. Provide multiple names separated by space."
  )

  parser.add_argument(
      "--runs", type=str, nargs='+', required=True,
      help="Run numbers for each model as space-separated strings, one per model "
           "(e.g., --runs \"1 2 3\" \"1 2\"). Must match the number of --models."
  )

  parser.add_argument(
      "--use_parallel", action="store_true", default=False,
      help="Run test evaluation in parallel using multiprocessing."
  )

  args = parser.parse_args()

  # Load problem_size and capacity from the training config
  project_root = Path(__file__).resolve().parent.parent
  problem_cfg_path = project_root / "cfg" / "problem" / f"{args.problem_name}.yaml"
  if not problem_cfg_path.exists():
    parser.error(f"Problem config not found: {problem_cfg_path}")
  with open(problem_cfg_path, "r") as f:
    problem_cfg = yaml.safe_load(f)

  problem_size = str(problem_cfg.get("problem_size", ""))
  if not problem_size:
    parser.error(f"'problem_size' not set in {problem_cfg_path}")

  raw_capacity = problem_cfg.get("capacity", None)
  if raw_capacity is not None:
    capacities = [int(c) for c in (raw_capacity if isinstance(raw_capacity, list) else [raw_capacity])]
  else:
    capacities = [None]

  # Parse per-model run numbers (one space-separated string per model)
  models_to_process = args.models
  if len(args.runs) != len(models_to_process):
    parser.error(
        f"Number of --runs strings ({len(args.runs)}) must match number of --models "
        f"({len(models_to_process)}). Provide one space-separated run string per model."
    )
  model_runs = {}
  for idx, model in enumerate(models_to_process):
    try:
      model_runs[model] = [int(r) for r in args.runs[idx].split()]
    except ValueError:
      parser.error(f"Invalid run numbers for model '{model}': '{args.runs[idx]}'. "
                   f"Expected space-separated integers.")

  print(args)
  print("Model runs:", model_runs)
  print(f"[+] problem_size={problem_size}, capacities={capacities} (loaded from {problem_cfg_path})")

  print("Starting evaluation...")
  problem_name = args.problem_name
  print(problem_name)
  problem_dir = Path("problems") / problem_name

  import sys
  if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
  helper = _import_helper(problem_name)
  dataset_dir = problem_dir / "dataset"

  print("Project root: ", project_root)

  results_root_dir = project_root / "results"
  print("Results root dir: ", results_root_dir)

  test_results_dir = project_root / "test_results"
  test_results_dir.mkdir(parents=True, exist_ok=True)
  print("Test results dir: ", test_results_dir)

  # Initialize cache for function scores across all runs
  score_cache = FunctionScoreCache()

  print(f"\n[+] ===== Problem Size: {problem_size} =====")
  print(f"[+] Loading dataset from: {dataset_dir} (mode=test)")

  if capacities != [None]:
    dataset_outputs = helper.read_dataset(dataset_dir, problem_size=problem_size, capacity=capacities, mode="test")
  else:
    dataset_outputs = helper.read_dataset(dataset_dir, problem_size=problem_size, mode="test")

  if isinstance(dataset_outputs, dict):
    dataset_dict = dataset_outputs
  else:
    print("Check input format")
  print(f"[+] Loaded {dataset_dict['n_instances']} instances")

  summary_path = test_results_dir / problem_name / "instance_counts.txt"
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  with open(summary_path, "w") as f:
    f.write(f"{problem_size}\t{dataset_dict['n_instances']}\n")
  print(f"[✓] Instance summary saved at: {summary_path}")

  # ========== EVALUATION LOOP ==========
  for model_name in args.models:
    print(f"\n[+] --- Evaluating Model: {model_name} ---")
    model_results_dir = results_root_dir / args.problem_name / model_name
    logs, loaded_runs = load_run_logs(model_results_dir, run_numbers=model_runs[model_name])
    print(f"[+] Loaded logs for runs: {loaded_runs}")
    best_programs = extract_training_data(logs, loaded_runs)

    for run in loaded_runs:
      print(f"\n[+] ---- Run {run} ----")
      programs_run = best_programs[run]
      results_dict = {}

      for call_id, (func_at_call, func_id) in programs_run.items():
        print(f"[.] Call {call_id}: func_id={func_id}")

        if func_at_call is None:
          print(f"[-] Skipping call {call_id} in run {run}: erroneous during training.")
          continue

        # ========== CACHE CHECK (deduplicates identical programs) ==========
        cached_result = score_cache.get(func_id)

        if cached_result is not None:
          print(f"[♻] Using cached result for func_id={func_id}")
          results_dict[call_id] = {
              "func_id": func_id,
              "func_code": func_at_call,
              "scores": cached_result['scores'],
              "avg_score": cached_result['avg_score'],
              "eval_time": cached_result['eval_time'],
              "eval_times": cached_result['eval_times'],
              "errors": cached_result['errors'],
              "timeouts": cached_result['timeouts'],
              "cached": True,
          }
          status = "✓ Valid" if cached_result['avg_score'] != -np.inf else "✗ Errored"
          print(f"[♻] Cached score: {cached_result['avg_score']:.2f} {status}")
        else:
          # ========== NEW EVALUATION ==========
          print(f"[→] Evaluating func_id={func_id} for the first time")

          priority_file = write_priority_func(problem_name, model_name, func_at_call)
          print(priority_file, type(priority_file))

          tester = Tester(dataset_dict, helper, priority_file)
          score_list, curr_avg_score, avg_eval_time, eval_times, errors, timeouts = tester.analyse(use_parallel=args.use_parallel)

          results_dict[call_id] = {
              "func_id": func_id,
              "func_code": func_at_call,
              "scores": score_list,
              "avg_score": curr_avg_score,
              "eval_time": avg_eval_time,
              "eval_times": eval_times,
              "errors": errors,
              "timeouts": timeouts,
              "cached": False,
          }

          score_cache.set(func_id, score_list, curr_avg_score, avg_eval_time, eval_times, errors, timeouts)

          status = "✓ Valid" if curr_avg_score != -np.inf else "✗ Errored"
          print(f"[+] New evaluation result: {curr_avg_score:.2f} {status}")

      out_path = test_results_dir / problem_name / model_name / f"test_res_{run}.pkl"
      os.makedirs(out_path.parent, exist_ok=True)
      with open(out_path, "wb") as f:
        pickle.dump(results_dict, f)
      print(f"[✓] Saved results for run {run} at {out_path}")

    create_excel_reports(problem_name, model_name, problem_size, loaded_runs, test_results_dir)

  # ========== CACHE STATISTICS ==========
  print(f"\n[+] Cache Statistics:")
  score_cache.print_stats()

  print(f"\n[✓] All evaluations complete!")
  print(f"[✓] Test results saved to: {test_results_dir}")
  print(f"[✓] Instance summary saved at: {summary_path}")
  # Final cache stats
  print(f"\n[+] Final Cache Statistics:")
  score_cache.print_stats()

if __name__ == "__main__":
  main()