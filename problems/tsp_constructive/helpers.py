from scipy.spatial import distance_matrix
import numpy as np
import ast
import textwrap
from os import path
from typing import Any
import sys
import os

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

def priority(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray, priority_file: Any) -> int:
  # print("In priority!!")
  with open(priority_file, 'r') as file:
    lines = file.read()

  functions = {}

  exec(lines, functions)

  for func_name, func in functions.items():
    if func_name.startswith('priority_v2') and callable(func):
      # print('Priority v2 executed !')
      result = func(current_node, destination_node, unvisited_nodes, distance_matrix)
      func_code = extract_function_code(lines, func_name)
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v1') and callable(func):
      # print('Priority v1 executed !')
      result = func(current_node, destination_node, unvisited_nodes, distance_matrix)
      func_code = extract_function_code(lines, func_name)
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v0') and callable(func):
      # print('Priority v0 executed !')
      result = func(current_node, destination_node, unvisited_nodes, distance_matrix)
      # print("Result: ", result)
      func_code = extract_function_code(lines, func_name)
      # print("Func Code: ", func_code)
      return result, func_name, func_code  

  return result, None, None


def solve(node_positions: np.ndarray, priority_file: Any) -> list:
    """
    Constructs a TSP tour using the priority heuristic.

    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).

    Returns
    -------
    tour : list[int]
        List of node indices representing the visiting order.
    """
    # print("In solve!!")
    problem_size = node_positions.shape[0]
    # print(problem_size)
    dist_mat = distance_matrix(node_positions, node_positions)

    start_node = 0
    tour = [start_node]
    unvisited = set(range(problem_size))
    unvisited.remove(start_node)

    for _ in range(problem_size - 1):
        # print("In loop!")
        # print(tour[-1], start_node, unvisited, dist_mat.shape)
        # x = unvisited
        # print("Below x")
        # y = dist_mat.copy()
        # print("Below y")
        next_node, func_name, func_code = priority(
            current_node=tour[-1],
            destination_node=start_node,
            unvisited_nodes=unvisited.copy(),
            distance_matrix=dist_mat.copy(),
            priority_file=priority_file
        )
        # print("next node: ", next_node)
        if next_node not in unvisited:
            # print("Hi")
            raise KeyError(f"Node {next_node} already visited.")

     
        tour.append(next_node)
        unvisited.remove(next_node)

    return tour, func_name, func_code

def evaluate(node_positions: np.ndarray, priority_file: Any) -> float:
    """
    Evaluates the TSP tour cost for given node positions.

    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).

    Returns
    -------
    cost : float
        Total length of the tour.
    """
    # print("In Eval!!!")
    # print(node_positions)
    dist_mat = distance_matrix(node_positions, node_positions)
    # print(dist_mat.shape)
    tour, func_name, func_code = solve(node_positions, priority_file)
    # print(tour)
    cost = 0.0
    for i in range(len(tour)):
        cost += dist_mat[tour[i], tour[(i + 1) % len(tour)]]
    return -cost, func_name, func_code

def read_dataset(basepath: str='.', problem_size: int=50, mode='train'):
    if(mode == 'train'):
        dataset_path = path.join(basepath, f"train{problem_size}_dataset.npy")
    elif(mode == 'test'):
        dataset_path = path.join(basepath, f"test{problem_size}_dataset.npy")
    elif(mode == 'val'):
        dataset_path = path.join(basepath, f"val{problem_size}_dataset.npy")
    else:
        print(f"[!] Invalid mode '{mode}'. Expected one of ['train', 'test', 'val'].")
        sys.exit(1)
    # print(dataset_path)
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset file not found: {dataset_path}")
        print("    Please check the basepath, problem_size, and mode arguments.")
        sys.exit(1)

    try:
        node_positions = np.load(dataset_path)
    except Exception as e:
        print(f"[!] Failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)

    n_instances = node_positions.shape[0]
    print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
    return {
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