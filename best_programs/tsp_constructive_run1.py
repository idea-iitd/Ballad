def priority_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
  """Improved version of `priority_v1`."""
  """
    This heuristic builds upon `priority_v1` by further refining the consideration
    of the path back to the destination. It introduces a factor that penalizes
    nodes that lead to a significantly longer path from *other unvisited nodes*
    to the destination. This encourages selecting nodes that contribute to a more
    efficient overall tour completion.

    The score for each unvisited node is calculated as follows:
    score = (1 - delta) * (distance to current_node)
            + delta * (distance to current_node - min_distance_to_any_unvisited)
            - alpha * (average distance from this node to all other unvisited nodes)
            - beta * (minimum distance from this node to any *other* unvisited node)
            - gamma * (distance from this node to destination_node)
            - epsilon * (average distance from *other unvisited nodes* to destination_node,
                         weighted by the distance from the current candidate node to those other unvisited nodes)

    Where:
    - `epsilon` is a weighting factor for the "destination path efficiency" heuristic.
      A higher `epsilon` more strongly penalizes nodes that lead to inefficient paths to the destination for the remaining nodes.
    - `delta`, `alpha`, `beta`, `gamma` are as defined in `priority_v1`.
    - `min_distance_to_any_unvisited` is the shortest distance from the
      `current_node` to any node in `unvisited_nodes`.

    The node with the *minimum* score is chosen.
  """
  scores = {}
  alpha = 0.4  # Weight for centrality
  beta = 0.3   # Weight for minimum distance to another unvisited node
  gamma = 0.3  # Weight for distance to destination
  delta = 0.2  # Weight for immediate greedy choice component
  epsilon = 0.1  # Weight for destination path efficiency

  if not unvisited_nodes:
      return destination_node # No nodes left to visit

  # Calculate the minimum distance from the current node to any unvisited node
  min_dist_current_to_unvisited = np.inf
  if unvisited_nodes:
      min_dist_current_to_unvisited = min(distance_matrix[current_node, node] for node in unvisited_nodes)

  unvisited_list = list(unvisited_nodes)
  num_unvisited = len(unvisited_list)

  # Pre-calculate average distances to other unvisited nodes
  average_distances_to_others = np.zeros(num_unvisited)
  if num_unvisited > 1:
      sub_distance_matrix = distance_matrix[np.ix_(unvisited_list, unvisited_list)]
      sum_distances_to_others = np.sum(sub_distance_matrix, axis=1) - np.diag(sub_distance_matrix)
      average_distances_to_others = sum_distances_to_others / (num_unvisited - 1)

  # Pre-calculate minimum distances to other unvisited nodes
  min_distances_to_others = np.full(num_unvisited, np.inf)
  if num_unvisited > 1:
      for i in range(num_unvisited):
          distances_from_node_i = np.delete(sub_distance_matrix[i, :], i)
          if distances_from_node_i.size > 0:
              min_distances_to_others[i] = np.min(distances_from_node_i)

  # Calculate the average distance from each unvisited node to the destination
  avg_distances_to_dest_from_unvisited = np.zeros(num_unvisited)
  if num_unvisited > 0:
      distances_from_unvisited_to_dest = np.array([distance_matrix[node, destination_node] for node in unvisited_list])
      if num_unvisited > 1:
          avg_distances_to_dest_from_unvisited = (np.sum(distances_from_unvisited_to_dest) - distances_from_unvisited_to_dest) / (num_unvisited - 1)
      else:
          avg_distances_to_dest_from_unvisited = np.zeros(num_unvisited) # No "other" nodes to average

  for i, node in enumerate(unvisited_nodes):
      distance_to_current = distance_matrix[current_node, node]
      avg_dist_to_remaining = average_distances_to_others[i] if num_unvisited > 0 else 0
      min_dist_to_other_unvisited = min_distances_to_others[i] if num_unvisited > 1 else 0
      distance_to_dest = distance_matrix[node, destination_node]

      # Calculate the destination path efficiency term
      destination_path_penalty = 0
      if num_unvisited > 1:
          # For each other unvisited node, calculate the distance from 'node' to that other node,
          # and multiply it by the average distance from those *other* unvisited nodes to the destination.
          # We want to penalize cases where 'node' is far from other unvisited nodes, and those other
          # unvisited nodes are themselves far from the destination.

          # Distances from the current candidate 'node' to all *other* unvisited nodes.
          distances_from_candidate_to_others = np.delete(sub_distance_matrix[i, :], i)

          # Average distances from all *other* unvisited nodes to the destination.
          # We need to carefully select these values.
          avg_dist_from_others_to_dest = np.delete(avg_distances_to_dest_from_unvisited, i)

          if distances_from_candidate_to_others.size > 0 and avg_dist_from_others_to_dest.size > 0:
              destination_path_penalty = np.sum(distances_from_candidate_to_others * avg_dist_from_others_to_dest) / (num_unvisited - 1)
              # We are incentivized to minimize this penalty, so we subtract it.

      # Combine heuristics. We want to minimize all these components.
      # A lower score is better.
      scores[node] = (
          (1 - delta) * distance_to_current
          + delta * (distance_to_current - min_dist_current_to_unvisited)
          - alpha * avg_dist_to_remaining
          - beta * min_dist_to_other_unvisited
          - gamma * distance_to_dest
          - epsilon * destination_path_penalty
      )

  if not scores:
      return destination_node # Should ideally not be reached

  # Find the node with the minimum score.
  next_node = min(scores, key=scores.get)
  return next_node