"""Generating node mappings between 2 labelled graphs such that edit distance is minimal"""
import numpy as np
import copy
import numpy as np
import ast
import textwrap
from typing import Any
import networkx as nx
import math
import heapq
import random
from dataclasses import dataclass
from typing import NamedTuple

class GEDInstance(NamedTuple):
    g1: list[list[int]]
    g2: list[list[int]]
    nf1: list[list[float]]
    nf2: list[list[float]]
    ged: int
    layer_val: int

def extract_function_code(source: str, func_name: str) -> str:
    tree = ast.parse(source)
    for node in reversed(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return textwrap.dedent(ast.get_source_segment(source, node))
    return f"# Could not extract code for {func_name}"

def convert_to_adjmatrix(edge_list1, edge_list2, max_node) -> list[list[int]]: 
  adjacency_matrix1 = [[0] * max_node for _ in range(max_node)]
  adjacency_matrix2 = [[0] * max_node for _ in range(max_node)]

  for edge in zip(edge_list1[0], edge_list1[1]):
    source, destination = edge
    adjacency_matrix1[source][destination] = 1
  
  for edge in zip(edge_list2[0], edge_list2[1]):
    source, destination = edge
    adjacency_matrix2[source][destination] = 1

  ged=2*max_node -2 - max(max(edge_list1[0]), max(edge_list1[1])) - max(max(edge_list2[0]), max(edge_list2[1]))
  return adjacency_matrix1, adjacency_matrix2, ged 

def assign_labels_based_on_features(nftrs1,nftrs2):
    labels = {}  # Dictionary to store unique feature-to-label mapping
    node_labels1 = []  # Dictionary to store node index to label mapping
    node_labels2 = []
    current_label_index = 0  # Tracks the next label to assign

    # Alphabet list for labeling
    alphabet = [chr(ord('a') + i) for i in range(26)]  # 'a', 'b', ..., 'z'
    
    # If more than 26 labels are needed, extend the alphabet (aa, ab, etc.)
    z=len(nftrs1)+len(nftrs2)
    while len(alphabet) < z :
        alphabet += [a + b for a in alphabet for b in alphabet][:z - len(alphabet)]
    

    for i, features in enumerate(nftrs1):
        features_tuple = tuple(features)  # Convert features to a tuple to make it hashable
        if features_tuple==(-1,):
           node_labels1.append("eps")
        else:
        # Check if features already have an assigned label
          if features_tuple in labels:
            node_labels1.append(labels[features_tuple])  # Assign existing label
          else:
              # Assign a new label
              labels[features_tuple] = alphabet[current_label_index]
              node_labels1.append(alphabet[current_label_index])
              current_label_index += 1

    for i, features in enumerate(nftrs2):
        features_tuple = tuple(features)  # Convert features to a tuple to make it hashable
        
        if features_tuple==(-1,):
           node_labels2.append("eps")
        # Check if features already have an assigned label
        else:
           if features_tuple in labels:
            node_labels2.append(labels[features_tuple])  # Assign existing label
           else:
            labels[features_tuple] = alphabet[current_label_index]
            node_labels2.append(alphabet[current_label_index])
            current_label_index += 1   

    return node_labels1, node_labels2

def priority(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]], priority_file: Any) -> list[list[float]]:
  # print("In priority!!")
  with open(priority_file, 'r') as file:
    lines = file.read()

  functions = {}
  
  # Execute the code in temp.txt
  exec(lines, functions)
  
  max_node = (max(max(graph1[0]), max(graph1[1]),max(graph2[0]),max(graph2[1])) + 1)
  result = [[0.0] * max_node for _ in range(max_node)]
  
  for func_name, func in functions.items():
    if func_name.startswith('priority_v2') and callable(func):
      # print('Priority v2 executed !')
      result = func(graph1, graph2, weights)
      func_code = extract_function_code(lines, func_name)
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v1') and callable(func):
    #   print('Priority v1 executed !')
      result = func(graph1, graph2, weights)
      func_code = extract_function_code(lines, func_name)
    #   print(f'\n{func_code}\n=================================================\n')
      return result, func_name, func_code

  for func_name, func in functions.items():
    if func_name.startswith('priority_v0') and callable(func):
      # print('Priority v0 executed !')
      result = func(graph1, graph2, weights)
      # print("Result: ", result)
      func_code = extract_function_code(lines, func_name)
      # print("Func Code: ", func_code)
      return result, func_name, func_code  

  return result, None, None

def solve(
    graph1: list[list[int]], 
    graph2: list[list[int]],
    labels1, labels2,
    weights: list[list[float]]
) -> dict:
    """Builds a mapping between 2 labelled graphs using Neighbor Bias Mapper and label similarity.
    
    Args:
        graph1: adjacency matrix of graph 1
        graph2: adjacency matrix of graph 2
        weights: A weight matrix representing the likelihood of mapping nodes between `graph1` and `graph2`.
    
    Returns:
        nodemappings: A dictionary mapping nodes of graph1 to graph2.
    """
    def label_similarity(u, v):
        return 1 if labels1[u] == labels2[v] else 0
    n1, n2 = len(graph1), len(graph2)

    
    # Initial priority weights adjusted with label similarity
    W=weights
    
    # Priority queue to store potential matches with highest priority first
    PQ = []
    mate = [-1] * n1  # stores best match in graph2 for each node in graph1
    wt = [-math.inf] * n1  # stores the best weight for each node in graph1

    # Initialize PQ with best initial matches
    for u in range(n1):
        v_m = max(range(n2), key=lambda v: W[u][v])  # best match for u in graph2
        heapq.heappush(PQ, (-W[u][v_m], u, v_m))  # store negative weight for max-heap behavior
        mate[u] = v_m
        wt[u] = W[u][v_m]

    matched = set()
    nodemappings = {}

    # Function to get neighbors up to 1 hop away
    def get_1hop_neighbors(u, graph):
        return [i for i in range(len(graph[u])) if graph[u][i] > 0]

    # Function to get neighbors up to 2 hops away
    def get_2hop_neighbors(u, graph):
        neighbors = set()
        for neighbor in range(len(graph[u])):
            if graph[u][neighbor] > 0:
                neighbors.add(neighbor)
                for second_neighbor in range(len(graph[neighbor])):
                    if graph[neighbor][second_neighbor] > 0:
                        neighbors.add(second_neighbor)
        return neighbors

    # Process the priority queue
    while PQ:
        _, u, v = heapq.heappop(PQ)
        if u in nodemappings:
            continue
        if v in matched:
            v_m = max((w for w in range(n2) if w not in matched), key=lambda x: W[u][x], default=None)
            if v_m is not None:
                heapq.heappush(PQ, (-W[u][v_m], u, v_m))
                mate[u] = v_m
                wt[u] = W[u][v_m]
            continue

        # Mark (u, v) as matched
        nodemappings[u] = v
        matched.add(v)

        # Update weights based on 1-hop neighbors
        neighbors_u = get_1hop_neighbors(u, graph1)
        neighbors_v = get_1hop_neighbors(v, graph2)
        
        for u_prime in neighbors_u:
            if u_prime in nodemappings:
                continue
            for v_prime in neighbors_v:
                if v_prime in matched:
                    continue
                # Update weight for u_prime and v_prime based on label similarity and 1-hop neighbors
                W[u_prime][v_prime] += W[u][v] + label_similarity(u_prime, v_prime)
                if W[u_prime][v_prime] > wt[u_prime]:
                    mate[u_prime] = v_prime
                    wt[u_prime] = W[u_prime][v_prime]
                    heapq.heappush(PQ, (-W[u_prime][v_prime], u_prime, v_prime))

    # Second pass: further refine weights using 2-hop neighbors
    PQ = []
    for u in range(n1):
        v_m = max(range(n2), key=lambda v: W[u][v])  # best match for u in graph2 after 1-hop update
        heapq.heappush(PQ, (-W[u][v_m], u, v_m))
        mate[u] = v_m
        wt[u] = W[u][v_m]

    # Process the priority queue again for 2-hop neighbors
    while PQ:
        _, u, v = heapq.heappop(PQ)
        if u in nodemappings:
            continue
        if v in matched:
            v_m = max((w for w in range(n2) if w not in matched), key=lambda x: W[u][x], default=None)
            if v_m is not None:
                heapq.heappush(PQ, (-W[u][v_m], u, v_m))
                mate[u] = v_m
                wt[u] = W[u][v_m]
            continue

        # Mark (u, v) as matched
        nodemappings[u] = v
        matched.add(v)

        # Update weights based on 2-hop neighbors
        neighbors_u = get_2hop_neighbors(u, graph1)
        neighbors_v = get_2hop_neighbors(v, graph2)
        
        for u_prime in neighbors_u:
            if u_prime in nodemappings:
                continue
            for v_prime in neighbors_v:
                if v_prime in matched:
                    continue
                # Update weight for u_prime and v_prime based on 2-hop neighbors
                W[u_prime][v_prime] += W[u][v] + label_similarity(u_prime, v_prime)
                if W[u_prime][v_prime] > wt[u_prime]:
                    mate[u_prime] = v_prime
                    wt[u_prime] = W[u_prime][v_prime]
                    heapq.heappush(PQ, (-W[u_prime][v_prime], u_prime, v_prime))

    return nodemappings

def evaluate(desc : GEDInstance, priority_file: Any):
  """Returns the graph edit distance based on the node mappings"""
  g1 = desc.g1
  g2 = desc.g2
  nf1 = desc.nf1
  nf2 = desc.nf2
  layer_val = desc.layer_val
  
  max_node = max(len(nf1),len(nf2))

  g1, g2, ged = convert_to_adjmatrix(g1,g2,max_node)
  # print('Max Node: ', max_node)
 
  x = len(g1[0])
  y1 = len(nf1)  # y is the number of nodes with features
  y2 = len(nf2)

  nf1 = nf1 + [[-1]] * (x - y1)
  nf2 = nf2 + [[-1]] * (x - y2)

  # for labelled
  labels1, labels2 = assign_labels_based_on_features(nf1,nf2) 

  weights = [[0.0] * max_node for _ in range(max_node)]
  for i in range(max_node):
        for j in range(max_node):
            # Handle dummy nodes (explicitly excluded from labels)
            if i >= len(labels1) or j >= len(labels2) or labels1[i] == "eps" or labels2[j] == "eps":
                weights[i][j] = 0.0
            elif labels1[i] == labels2[j]:
                weights[i][j] = 1.0
            elif labels1[i] != labels2[j]:
                weights[i][j] = 0.0
  #normalise
  for i in range(max_node):
        row_sum = sum(weights[i][:max_node]) # Only normalize within valid range of n2
        if row_sum > 0:
            for j in range(max_node):
                weights[i][j] /= row_sum

  wts, func_name, func_code  = priority(copy.deepcopy(g1), copy.deepcopy(g2), weights, priority_file)

  if wts is not None:
    for _ in range(layer_val): 
      x, func_name, func_code = priority(copy.deepcopy(g1), copy.deepcopy(g2), wts, priority_file)
      if x is None:
        break
      wts=x
  
  nodemap = solve(copy.deepcopy(g1),copy.deepcopy(g2), copy.deepcopy(labels1), copy.deepcopy(labels2), wts)  
  # print("Nodemap:", nodemap)
  cost = 0
  y=0
  
  for v1 in range(len(g1)):
     mapped_v1 = nodemap[v1]
     if labels1[v1]!="eps" and labels2[mapped_v1]!="eps" and labels1[v1]!=labels2[mapped_v1]:
        y+=1

  for v1 in range(len(g1)):
      for v2 in range(len(g1[0])):
          mapped_v1 = nodemap.get(v1, -1)
          mapped_v2 = nodemap.get(v2, -1)
          if g1[v1][v2] == 1:
              if mapped_v1 != -1 and mapped_v2 != -1 and g2[mapped_v1][mapped_v2] != 1:
                  cost += 1

          elif g1[v1][v2] == 0:

              if mapped_v1 != -1 and mapped_v2 != -1 and g2[mapped_v1][mapped_v2] != 0:
                  cost += 1

  cost = cost/2+ ged+ y

  print("Nodemap:", nodemap)
  print("Cost:", cost)
  return -cost, func_name, func_code  

# def process_file(file_path, layer_val):
#   with open(file_path, 'r') as file:
#     content = file.read()
#     g_pairs_str = content.split("g_pairs: ")[1].split("node_features: ")[0].strip()
#     nf_str = content.split("node_features: ")[1].split("ground_truth: ")[0].strip()
#     ground_truth_str = content.split("ground_truth: ")[1].strip()
#     g_pairs = ast.literal_eval(g_pairs_str)
#     nfs = ast.literal_eval(nf_str)
#     ground_truth = ast.literal_eval(ground_truth_str)
#     data = []
#     for i, (g_pair, nfs, ged) in enumerate(zip(g_pairs, nfs, ground_truth)):
#       # Assume g_pair is of the form [G1, G2] and node features are extracted as nf1 and nf2
#       G1, G2 = g_pair
#       nf1, nf2 = nfs  # Replace with actual logic for node features if available
#       data.append(GEDInstance(G1, G2, nf1, nf2, ged, layer_val))  
#   return data, ground_truth

def process_file(file_path, layer_val):
  with open(file_path, 'r') as file:
    content = file.read()
    has_ground_truth = "ground_truth: " in content
    g_pairs_str = content.split("g_pairs: ")[1].split("node_features: ")[0].strip()
    nf_section = content.split("node_features: ")[1]
    nf_str = nf_section.split("ground_truth: ")[0].strip() if has_ground_truth else nf_section.strip()
    g_pairs = ast.literal_eval(g_pairs_str)
    nfs = ast.literal_eval(nf_str)
    ground_truth = ast.literal_eval(content.split("ground_truth: ")[1].strip()) if has_ground_truth else [None] * len(g_pairs)
    data = []
    for i, (g_pair, nf, ged) in enumerate(zip(g_pairs, nfs, ground_truth)):
      G1, G2 = g_pair
      nf1, nf2 = nf
      data.append(GEDInstance(G1, G2, nf1, nf2, ged, layer_val))
  return data, ground_truth

def read_dataset(basepath: str='.', dataset_name: str='linux', mode: str="train", layer_val: int=0):
    import sys
    import os
    
    # Validate mode
    valid_modes = {"train", "test", "val"}
    if mode not in valid_modes:
        print(f"[!] Invalid mode '{mode}'. Expected one of {sorted(valid_modes)}.")
        sys.exit(1)

    data_dir = basepath / "data" / f"{mode}"
    suff_dict = {"train": "", "test": "_test", "val": "_val_1k"}
    dataset_path = os.path.join(data_dir, f"{dataset_name}{suff_dict[mode]}.txt")

    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset file not found: {dataset_path}")
        print("    Please check the basepath, dataset_name, and mode arguments.")
        sys.exit(1)

    # Try loading dataset safely
    try:
        dataset, ground_truth = process_file(dataset_path, layer_val)
    except Exception as e:
        print(f"[!] Failed to load dataset from {dataset_path}: {e}")
        sys.exit(1)

    n_instances = len(dataset)
    print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

    return {
        "dataset": dataset,
        "ground_truth": ground_truth,
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
        # Bin packing and GED
        return dataset_dict["dataset"]
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_dict.keys()}")
