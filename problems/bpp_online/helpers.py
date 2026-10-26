import numpy as np
import ast
import textwrap
from os import path
from typing import Any, List
import pickle

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

def priority(item: float, bins_remain_cap: np.ndarray, priority_file: Any) -> np.ndarray:
    with open(priority_file, 'r') as file:
        lines = file.read()

    functions = {}
    exec(lines, functions)

    for prefix in ['priority_v2', 'priority_v1', 'priority_v0']:
        for func_name, func in functions.items():
            if func_name.startswith(prefix) and callable(func):
                func_code = extract_function_code(lines, func_name)
                try:
                    result = func(item, bins_remain_cap)
                except Exception as e:
                    # If timeout or runtime error occurs, return safe placeholders
                    # Result can be None or a neutral priority vector depending on your logic
                    result = None
                    # Optional: log or print for debugging
                    print(f"{func_name} execution failed: {e}")
                return result, func_name, func_code

    # If nothing matched
    return None, None, None

#   # print("In priority!!")
#   with open(priority_file, 'r') as file:
#     lines = file.read()

#   functions = {}

#   exec(lines, functions)

#   for func_name, func in functions.items():
#     if func_name.startswith('priority_v2') and callable(func):
#       # print('Priority v2 executed !')
#       result = func(item, bins_remain_cap)
#       func_code = extract_function_code(lines, func_name)
#       return result, func_name, func_code

#   for func_name, func in functions.items():
#     if func_name.startswith('priority_v1') and callable(func):
#       # print('Priority v1 executed !')
#       result = func(item, bins_remain_cap)
#       func_code = extract_function_code(lines, func_name)
#       return result, func_name, func_code

#   for func_name, func in functions.items():
#     if func_name.startswith('priority_v0') and callable(func):
#       # print('Priority v0 executed !')
#       result = func(item, bins_remain_cap)
#       # print("Result: ", result)
#       func_code = extract_function_code(lines, func_name)
#       # print("Func Code: ", func_code)
#       return result, func_name, func_code  

#   return result, None, None



def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which the item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def solve(items: np.ndarray, capacity: float, priority_file: Any):
    """
    Performs online bin packing using the provided priority heuristic.

    Args:
        items : np.ndarray
            1D array of item sizes to be packed.
        capacity : float
            Capacity of each bin.

    Returns:
        packing : list[list[float]]
            List of bins, each containing packed items.
    """
    num_items = len(items)
    bins = np.array([capacity for _ in range(num_items)], dtype=float)
    packing = [[] for _ in bins]

    func_name, func_code = None, None

    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        if len(valid_bin_indices) == 0:
            continue
        priorities, func_name, func_code = priority(item, bins[valid_bin_indices], priority_file)
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)

    used_bins = [bin_items for bin_items in packing if len(bin_items)>0]
    return used_bins, func_name, func_code

# def evaluate(instance: dict, priority_file: Any) -> float:
#     """
#     Evaluates the heuristic on a single BPP instance.

#     Args:
#         instance : dict
#             Dictionary with keys 'capacity', 'items', and 'num_items'.
#         priority_file : Any
#             Path to the Python file containing the priority function.

#     Returns:
#         score : float
#             Negative of the number of bins used (to minimize bin count).
#     """
#     packing, func_name, func_code = solve(
#         np.array(instance["items"], dtype=float),
#         instance["capacity"],
#         priority_file
#     )
#     return -len(packing), func_name, func_code

# def read_dataset(basepath: str = ".", problem_size: str = '5k', mode: str = "train"):
#     """
#     Reads and loads the BPP dataset from pickle files.

#     Args:
#         basepath : str
#             Base directory where dataset files are stored.
#         problem_size : int
#             Problem size for which dataset should be loaded.

#     Returns:
#         dict
#             A dictionary containing:
#             - 'dataset': all loaded instances
#             - 'n_instances': total number of instances
#     """
#     # dataset_dir = path.join(basepath, "dataset")
#     print(basepath)
#     dataset_dir = path.join(basepath, "problems", "bpp_online", "dataset")
#     i=2 #i=1 for capacity 100, i=2 for capacity 500
#     fname = f"weibull_{problem_size}_{mode}{i}.pickle"                
#     dataset_path = path.join(dataset_dir, fname)
#     with open(dataset_path, "rb") as f:
#         data = pickle.load(f)

#     n_instances = len(data)
#     print(f"[*] Loaded {n_instances} dataset files from {dataset_dir}")
#     print(data)
#     print(len(data['train_0']['items']))
#     return {
#         "dataset": data,
#         "l1_bound": data['l1_bound'] if 'l1_bound' in data else None,
#         "n_instances": n_instances,
#     }

def is_valid_packing(packing: list[list[float]], instance: dict) -> bool:
    """
    Checks whether `packing` is a valid packing for the given instance.

    A packing is valid if:
    - Every item in instance["items"] appears exactly once across all bins.
    - No bin exceeds instance["capacity"].

    Args:
        packing : list[list[float]]
            List of bins, where each bin is a list of item sizes.
        instance : dict
            Dictionary with at least 'items' and 'capacity' keys.

    Returns:
        bool : True if packing is valid, False otherwise.
    """
    items = instance["items"]
    capacity = instance["capacity"]

    # Flatten all packed items
    packed_items = [x for bin_items in packing for x in bin_items]

    # Check all items are packed exactly once
    if sorted(packed_items) != sorted(items):
        return False

    # Check that no bin exceeds capacity
    for bin_items in packing:
        if sum(bin_items) > capacity + 1e-8:  # tolerance for floating-point precision
            return False

    return True


def evaluate(instance: dict, priority_file: any) -> tuple[float, str, str]:
    """
    Evaluates the heuristic on a single BPP instance.

    Args:
        instance : dict
            Dictionary with keys 'capacity', 'items', 'num_items', and 'l1_bound'.

    Returns:
        tuple
            (excess_percentage, func_name, func_code)
            where:
            - excess_percentage = ((num_bins - l1_bound) / l1_bound) * 100
                (lower is better)
    """
    packing, func_name, func_code = solve(
        np.array(instance["items"], dtype=float),
        instance["capacity"],
        priority_file
    )

    if not is_valid_packing(packing, instance):
        print(f"[!] Invalid packing detected for function {func_name}.")
        return float("-inf"), func_name, func_code  # penalize invalid heuristics heavily
    
    num_bins = len(packing)
    l1_bound = instance.get("l1_bound", None)

    if l1_bound is not None and l1_bound > 0:
        excess = ((num_bins - l1_bound) / l1_bound) * 100
    else:
        excess = float("inf")  # fallback if bound missing or invalid

    print("L1 Bound: ", l1_bound)
    print("Num bins: ", num_bins)
    print("Excess: ", excess)
    
    return -excess, func_name, func_code

# def read_dataset(basepath: str = ".", problem_size: str = "5k", mode: str = "train"):
#     """
#     Reads and loads the BPP dataset from two pickle files (capacities 100 and 500).

#     Args:
#         basepath : str
#             Base directory where dataset files are stored.
#         problem_size : str
#             Problem size for which dataset should be loaded.
#         mode : str
#             Dataset mode, e.g., 'train', 'val', or 'test'.

#     Returns:
#         dict
#             A dictionary containing:
#             - 'dataset': merged instances from both capacity files,
#               each with its own l1_bound
#             - 'n_instances': total number of instances
#     """
#     # print(f"Dataset directory (helpers): {basepath}")
#     data_combined = [] #{}
#     total_instances = 0

#     for i in [1, 2]:  # i=1 -> capacity 100, i=2 -> capacity 500
#         fname = f"weibull_{problem_size}_{mode}{i}.pickle"
#         dataset_path = path.join(basepath, fname)

#         with open(dataset_path, "rb") as f:
#             data = pickle.load(f)

#         l1_bound = data.get("l1_bound", None)

#         # add each instance, tagging it with its capacity and l1_bound
#         for key, val in data.items():
#             if key == "l1_bound":
#                 continue
#             # new_key = f"{key}_cap{val['capacity']}"
#             val["l1_bound"] = l1_bound  # attach its own bound
#             # data_combined[new_key] = val
#             data_combined.append(val)
#             total_instances += 1

#     print(f"[*] Loaded {total_instances} total instances from {basepath}")
#     # print(data_combined)
#     return {
#         "dataset": data_combined,
#         "n_instances": total_instances,
#     }

def read_dataset(basepath: str = ".", problem_size: str = "5k", capacity: List[int] = [100,500], mode: str = "train"):
    """
    Reads and loads the BPP dataset from two pickle files.

    For 'train' and 'val' modes, files are:
        weibull_{problem_size}_{mode}1.pickle
        weibull_{problem_size}_{mode}2.pickle

    For 'test' mode, files are:
        weibull_{problem_size}_{mode}100.pickle
        weibull_{problem_size}_{mode}500.pickle

    Args:
        basepath : str
            Base directory where dataset files are stored.
        problem_size : str
            Problem size for which dataset should be loaded.
        mode : str
            Dataset mode, e.g., 'train', 'val', or 'test'.

    Returns:
        dict
            {
                "dataset": list of merged instances (each tagged with its own l1_bound),
                "n_instances": total number of instances
            }
    """
    import sys
    import os
    
    valid_modes = {"train", "val", "test"}
    if mode not in valid_modes:
        print(f"[!] Invalid mode '{mode}'. Expected one of {sorted(valid_modes)}.")
        sys.exit(1)

    # Select suffixes depending on mode
    suffixes = []
    for cap in capacity:
        if mode in {"train", "val"}:
            if cap == 100:
                suffixes.append(1)
            elif cap == 500:
                suffixes.append(2)
        else:
            if cap == 100:
                suffixes.append(100)
            elif cap == 500:
                suffixes.append(500)
    # suffixes = [1, 2] if mode in {"train", "val"} else [100, 500]

    data_combined = []
    total_instances = 0

    for i in suffixes:
        if mode in {"train", "val"}:
            fname = f"weibull_{problem_size}_{mode}{i}.pickle"
        else:
            fname = f"weibull_{problem_size}_{mode}_{i}.pickle"
        dataset_path = os.path.join(basepath, fname)

        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"[!] Dataset file not found: {dataset_path}")
            print("    Please check the basepath, problem_size, and mode arguments.")
            sys.exit(1)

        # Try loading dataset
        try:
            with open(dataset_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[!] Failed to load dataset from {dataset_path}: {e}")
            sys.exit(1)

        l1_bound = data.get("l1_bound", None)

        # Add each instance, tagging with its own bound
        for key, val in data.items():
            if key == "l1_bound":
                continue
            val["l1_bound"] = l1_bound
            data_combined.append(val)
            total_instances += 1

    print(f"[*] Loaded {total_instances} total instances from {basepath}")
    return {
        "dataset": data_combined,
        "n_instances": total_instances,
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