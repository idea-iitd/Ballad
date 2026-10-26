from scipy.spatial import distance_matrix
import numpy as np
import ast
import textwrap
import torch
from .aco import ACO, BatchedACO
from typing import Any
import inspect


def _get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# def extract_function_code(source: str, func_name: str) -> str:
#     tree = ast.parse(source)
#     for node in tree.body:
#         if isinstance(node, ast.FunctionDef) and node.name == func_name:
#             return textwrap.dedent(ast.get_source_segment(source, node))
#     return f"# Could not extract code for {func_name}"

def extract_function_code(source: str, func_name: str) -> str:
    tree = ast.parse(source)
    for node in reversed(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return textwrap.dedent(ast.get_source_segment(source, node))
    return f"# Could not extract code for {func_name}"

def priority(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int, priority_file: Any) -> np.ndarray:
  # print("In priority!!")
  with open(priority_file, 'r') as file:
    lines = file.read()

  functions = {}
#   import os
#   import sys
#   print(os.getcwd())
#   print(sys.path)
  # Execute the code in temp.txt
  exec(lines, functions)
#   print("From helpers.priority: \n", lines, "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")  

  for func_name, func in functions.items():
    if func_name.startswith('priority_v2') and callable(func):
      # print('Priority v2 executed !')
      result = func(distance_matrix, coordinates, demands, capacity)
      func_code = extract_function_code(lines, func_name)
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v1') and callable(func):
      print('Priority v1 executed !')
    #   try:
    #         source = inspect.getsource(func)
    #         print("Function body:\n")
    #         print(source)
    #   except OSError:
    #     print("Source code not available for this function.")
      result = func(distance_matrix, coordinates, demands, capacity)
      func_code = extract_function_code(lines, func_name)
      print(f'\n{func_code}\n=================================================\n')
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v0') and callable(func):
      # print('Priority v0 executed !')
      result = func(distance_matrix, coordinates, demands, capacity)
      # print("Result: ", result)
      func_code = extract_function_code(lines, func_name)
      # print("Func Code: ", func_code)
      return result, func_name, func_code  

  return result, None, None

def solve(node_positions: np.ndarray, demand: np.ndarray, priority_file: Any) -> float:
    """
    Solves the Capacitated Vehicle Routing Problem (CVRP) using Ant Colony Optimization (ACO).

    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (n_nodes, 2).
    demand : np.ndarray
        1D array of length n_nodes representing demand at each node.

    Returns
    -------
    obj : float
        Objective value (e.g., total distance of routes).
    """
    N_ITERATIONS = 100
    N_ANTS = 30
    CAPACITY = 50

    # Compute distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    dist_mat[np.diag_indices_from(dist_mat)] = 1  # avoid zero distances on diagonal

    # Construct heuristic information
    heu, func_name, func_code = priority(dist_mat.copy(), node_positions.copy(), demand.copy(), CAPACITY, priority_file) #+ 1e-9
    heu = heu + 1e-9
    print('Hue: ', heu.shape, func_name)
    heu[heu < 1e-9] = 1e-9  # numerical stability

    # Run Ant Colony Optimization
    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)

    return obj, func_name, func_code

def evaluate(node_positions: np.ndarray, demand: np.ndarray, priority_file: Any) -> float:
    """
    Evaluates the CVRP solution cost for given node positions and demands.

    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (n_nodes, 2).
    demand : np.ndarray
        1D array of length n_nodes representing demand at each node.

    Returns
    -------
    cost : float
        Total length of the vehicle routes.
    """
    print(node_positions.shape, demand.shape)
    obj, func_name, func_code = solve(node_positions, demand, priority_file)
    return -obj.item(), func_name, func_code


def _load_priority_fn(priority_file: Any):
    """Load a priority file once and return (priority_fn, func_name, func_code)."""
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


def solve_batch(node_positions_list: list, demand_list: list, priority_file: Any):
    """
    Solve a list of CVRP instances as one GPU-batched ACO pass.

    Priority matrices are computed per-instance on CPU (LLM-generated numpy code),
    then stacked into (B,n,n) tensors and fed to BatchedACO which runs all B
    instances simultaneously on the best available device (CUDA > MPS > CPU).

    Returns
    -------
    costs : np.ndarray, shape (B,)   — lowest route length per instance
    func_name : str
    func_code : str
    """
    N_ITERATIONS = 100
    N_ANTS = 30
    CAPACITY = 50
    device = _get_device()

    priority_fn, func_name, func_code = _load_priority_fn(priority_file)
    if priority_fn is None:
        raise RuntimeError(f"No priority_v* function found in {priority_file}")

    heu_list, dist_list = [], []
    for node_positions, demand in zip(node_positions_list, demand_list):
        dist_mat = distance_matrix(node_positions, node_positions)
        dist_mat[np.diag_indices_from(dist_mat)] = 1

        heu = priority_fn(dist_mat.copy(), node_positions.copy(), demand.copy(), CAPACITY)
        heu = np.maximum(heu + 1e-9, 1e-9)

        dist_list.append(dist_mat)
        heu_list.append(heu)

    dist_t = torch.tensor(np.stack(dist_list),   dtype=torch.float32, device=device)
    heu_t  = torch.tensor(np.stack(heu_list),    dtype=torch.float32, device=device)
    dem_t  = torch.tensor(np.stack(demand_list), dtype=torch.float32, device=device)

    aco   = BatchedACO(dist_t, dem_t, heu_t, CAPACITY, n_ants=N_ANTS, device=device)
    costs = aco.run(N_ITERATIONS).cpu().numpy()   # (B,)
    return costs, func_name, func_code


def evaluate_batch(inputs: list, priority_file: Any):
    """
    Batch GPU evaluation for CVRP.

    inputs : list of (node_positions, demands) tuples — same format as get_inputs() returns.
    Returns (-costs, func_name, func_code) where higher score is better.
    """
    node_positions_list = [inp[0] for inp in inputs]
    demand_list         = [inp[1] for inp in inputs]
    costs, func_name, func_code = solve_batch(node_positions_list, demand_list, priority_file)
    return -costs, func_name, func_code


# def read_dataset(basepath: str='.', problem_size: int=50, mode: str="train"):
#     import os
#     dataset_path = os.path.join(basepath, f"{mode}{problem_size}_dataset.npy")
#     dataset = np.load(dataset_path)
#     demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
#     n_instances = node_positions.shape[0]
#     return {
#        "demands": demands,
#         "node_positions": node_positions,
#         "n_instances": n_instances,
#     }

def read_dataset(basepath: str='.', problem_size: int=50, mode: str="train"):
    import sys
    import os
    # Validate mode
    valid_modes = {"train", "test", "val"}
    if mode not in valid_modes:
        print(f"[!] Invalid mode '{mode}'. Expected one of {sorted(valid_modes)}.")
        sys.exit(1)

    dataset_path = os.path.join(basepath, f"{mode}{problem_size}_dataset.npy")

    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset file not found: {dataset_path}")
        print("    Please check the basepath, problem_size, and mode arguments.")
        sys.exit(1)

    # Try loading dataset safely
    try:
        dataset = np.load(dataset_path)
    except Exception as e:
        print(f"[!] Failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)

    # Extract data
    try:
        demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
    except Exception as e:
        print(f"[!] Dataset format error in {dataset_path}: {e}")
        sys.exit(1)

    n_instances = node_positions.shape[0]
    print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

    return {
        "demands": demands,
        "node_positions": node_positions,
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