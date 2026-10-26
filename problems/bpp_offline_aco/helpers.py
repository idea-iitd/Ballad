import ast
import functools
import textwrap
import numpy as np
import torch
from .gen_inst import BPPInstance, load_dataset
from .aco import ACO, BatchedACO, _get_device, reset_uniform_generator
from typing import Any

# def extract_function_code(source: str, func_name: str) -> str:
#     tree = ast.parse(source)
#     for node in tree.body:
#         if isinstance(node, ast.FunctionDef) and node.name == func_name:
#             return textwrap.dedent(ast.get_source_segment(source, node))
#     return f"# Could not extract code for {func_name}"

def extract_function_code(source: str, func_name: str) -> str:
    try:
        tree = ast.parse(source)
    except:
        return f"# From helpers: Could not extract code for {func_name}"
        exit(0)
    for node in reversed(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return textwrap.dedent(ast.get_source_segment(source, node))
    return f"# Could not extract code for {func_name}"

def priority(demand: np.ndarray, capacity: int, priority_file: Any) -> np.ndarray:
  # print("In priority!!")
  with open(priority_file, 'r') as file:
    lines = file.read()

  functions = {}
  exec(lines, functions)
    
  for func_name, func in functions.items():
    if func_name.startswith('priority_v2') and callable(func):
      # print('Priority v2 executed !')
      result = func(demand, capacity)
      func_code = extract_function_code(lines, func_name)
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v1') and callable(func):
      # print('Priority v1 executed !')
      result = func(demand, capacity)
      func_code = extract_function_code(lines, func_name)
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v0') and callable(func):
      # print('Priority v0 executed !')
      result = func(demand, capacity)
      # print("Result: ", result)
      func_code = extract_function_code(lines, func_name)
      # print("Func Code: ", func_code)
      return result, func_name, func_code  

  return result, None, None

def solve(inst: "BPPInstance", mode: str = 'sample', priority_file: Any = 'gemini-2.5-flash-lite') -> float:
    """
    Solves the Bin Packing Problem (BPP) using Ant Colony Optimization (ACO).

    Parameters
    ----------
    inst : BPPInstance
        Problem instance containing:
          - n : int
              Number of items.
          - capacity : int
              Capacity of each bin.
          - demands : np.ndarray
              1D array of item demands of shape (n,). Each entry represents 
              the demand (or size) of an item.
    mode : str, optional (default = 'sample')
        Mode of execution:
          - 'sample' : Run sampling only for a fixed number of solutions.
          - otherwise: Run full ACO optimization.

    Returns
    -------
    obj : float
        Objective value (e.g., number of bins used).
    """
    N_ITERATIONS = 15
    N_ANTS = 20
    SAMPLE_COUNT = 200

    reset_uniform_generator()  # flush stale pre-fetched buffer so np.random.seed takes effect
    priority_fn, func_name, func_code = _load_priority_fn(str(priority_file))
    heu = priority_fn(inst.demands.copy(), inst.capacity)
    assert tuple(heu.shape) == (inst.n, inst.n)
    assert 0 < heu.max() < np.inf

    aco = ACO(
        inst.demands,
        heu.astype(float),
        capacity=inst.capacity,
        n_ants=N_ANTS,
        greedy=False
    )

    if mode == 'sample':
        obj, _ = aco.sample_only(SAMPLE_COUNT)
    else:
        obj, _ = aco.run(N_ITERATIONS)

    return obj, func_name, func_code

def evaluate(inst: "BPPInstance", priority_file: Any) -> float:
    """
    Evaluates the BPP solution cost for a given problem instance.

    Parameters
    ----------
    inst : BPPInstance
        Problem instance containing:
          - n : int
              Number of items.
          - capacity : int
              Capacity of each bin.
          - demands : np.ndarray
              1D array of item demands of shape (n,).

    Returns
    -------
    cost : float
        Number of bins used.
    """
    obj, func_name, func_code = solve(inst, mode='full', priority_file=priority_file)#'sample
    return -obj, func_name, func_code


# @functools.lru_cache(maxsize=16)
def _load_priority_fn(priority_file: str):
    """Load a priority file once and cache by path. Returns (priority_fn, func_name, func_code)."""
    with open(priority_file, 'r') as f:
        lines = f.read()
    functions = {}
    exec(lines, functions)  # noqa: S102
    for prefix in ('priority_v2', 'priority_v1', 'priority_v0'):
        for name, fn in functions.items():
            if name.startswith(prefix) and callable(fn):
                func_code = extract_function_code(lines, name)
                return fn, name, func_code
    return None, None, None


def solve_batch(instances: list, priority_file: Any):
    """
    Solve a list of BPPInstances as one GPU-batched ACO pass.

    The priority function (LLM-generated, numpy) is called once per instance on CPU,
    then all heuristics are stacked into a (B,n,n) tensor and fed to BatchedACO which
    runs all instances simultaneously on the best available device (CUDA > MPS > CPU).

    Returns
    -------
    costs    : np.ndarray of shape (B,)  — number of bins per instance
    func_name: str
    func_code: str
    """
    N_ITERATIONS = 15
    N_ANTS = 20
    device = _get_device()

    priority_fn, func_name, func_code = _load_priority_fn(priority_file)
    if priority_fn is None:
        raise RuntimeError(f"No priority_v* function found in {priority_file}")

    heu_list = []
    capacity = instances[0].capacity  # same for all instances in a dataset

    for inst in instances:
        heu = priority_fn(inst.demands.copy(), inst.capacity)
        heu = np.clip(heu, 1e-6, 1e6).astype(float)
        heu = heu / heu.max()
        heu = np.clip(heu, 1e-6, None)
        heu_list.append(heu)

    demands_list = [inst.demands for inst in instances]

    dem_t = torch.tensor(np.stack(demands_list), dtype=torch.long, device=device)
    heu_t = torch.tensor(np.stack(heu_list),     dtype=torch.float32, device=device)

    aco   = BatchedACO(dem_t, heu_t, capacity, n_ants=N_ANTS, device=device)
    costs = aco.run(N_ITERATIONS).cpu().numpy()   # (B,) float — number of bins

    return costs, func_name, func_code


def evaluate_batch(inputs: list, priority_file: Any):
    """Batch GPU evaluation; returns (-costs, func_name, func_code) — higher score is better."""
    costs, func_name, func_code = solve_batch(inputs, priority_file)
    return -costs, func_name, func_code


# def read_dataset(basepath: str='.', problem_size: int=500, mode: str="train"):
#     import os
#     dataset_path = os.path.join(basepath, f"{mode}{problem_size}_dataset.npz")
#     dataset = load_dataset(dataset_path)
#     n_instances = len(dataset)
#     print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

#     return {
#        "dataset": dataset,
#         "n_instances": n_instances,
#     }

def read_dataset(basepath: str='.', problem_size: int=500, mode: str="train"):
    """
    Reads and loads a dataset stored in .npz format.

    Args:
        basepath : str
            Base directory where dataset files are stored.
        problem_size : int
            Problem size for which dataset should be loaded.
        mode : str
            Dataset mode, e.g., 'train', 'val', or 'test'.

    Returns:
        dict
            {
                "dataset": loaded dataset (as returned by load_dataset),
                "n_instances": total number of instances
            }
    """
    import sys
    import os
    # Validate mode
    valid_modes = {"train", "val", "test"}
    if mode not in valid_modes:
        print(f"[!] Invalid mode '{mode}'. Expected one of {sorted(valid_modes)}.")
        sys.exit(1)

    dataset_path = os.path.join(basepath, f"{mode}{problem_size}_dataset.npz")

    # Check file existence
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset file not found: {dataset_path}")
        print("    Please check the basepath, problem_size, and mode arguments.")
        sys.exit(1)

    # Try loading dataset
    try:
        dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"[!] Failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)

    n_instances = len(dataset)
    print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

    return {
        "dataset": dataset,
        "n_instances": n_instances,
    }

def get_inputs(dataset_dict):
    """
    Extracts the actual inputs that Evaluator should iterate over.
    """
    if "node_positions" in dataset_dict and "demands" in dataset_dict:
        # CVRP case
        return list(zip(dataset_dict["node_positions"], dataset_dict["demands"]))
    elif "node_positions" in dataset_dict:
        # TSP case
        return dataset_dict["node_positions"]
    elif "dataset" in dataset_dict:
        # Bin packing
        return dataset_dict["dataset"]
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_dict.keys()}")
