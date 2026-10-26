def priority_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
  """Improved version of `priority_v1`."""
  """
    Improved version of `priority_v2`.

    This heuristic aims to balance the immediate cost of traveling to a candidate node
    with a prediction of the *remaining tour cost* if that node is chosen. It uses the
    concept of the Nearest Neighbor heuristic for the remaining unvisited nodes, but
    instead of just picking the closest, it calculates the cost of completing the tour
    from each unvisited node.

    The scoring function for a candidate `node` is:
    `score = distance_matrix[current_node, node] + heuristic_remaining_cost(node)`

    Where:
    - `distance_matrix[current_node, node]`: The cost to travel to the candidate node.
    - `heuristic_remaining_cost(node)`: This is an estimation of the cost to visit all
                                         other unvisited nodes and return to the destination,
                                         starting from `node`. This is approximated by a
                                         simplified Nearest Neighbor-like traversal from `node`
                                         through the remaining `unvisited_nodes - {candidate_node}`
                                         and finally to `destination_node`.

    A lower score indicates a more desirable node.

    Args:
        current_node : int
            The index of the node currently being visited.
        destination_node : int
            The index of the tour's starting node (used for returning at the end of the tour).
        unvisited_nodes : set[int]
            The set of node indices that have not yet been visited.
        distance_matrix : np.ndarray
            A symmetric 2D array of shape (n, n) where entry [i, j] represents
            the Euclidean distance between node `i` and node `j`.

    Returns:
        int
            The index of the next node to visit, selected from `unvisited_nodes`
            based on this remaining tour cost heuristic.
  """
  scores = {}

  if not unvisited_nodes:
      return -1 # Indicate no node can be selected

  for candidate_node in unvisited_nodes:
      # Cost to travel to the candidate node
      immediate_cost = distance_matrix[current_node, candidate_node]

      # Estimate the cost of the remaining tour if candidate_node is chosen
      remaining_unvisited = unvisited_nodes - {candidate_node}
      estimated_remaining_cost = 0

      if remaining_unvisited:
          # Simulate a greedy path from candidate_node to visit all remaining
          # unvisited nodes and finally to the destination.
          current_heuristic_node = candidate_node
          temp_unvisited = copy(remaining_unvisited)

          while temp_unvisited:
              # Find the nearest neighbor among the remaining unvisited nodes
              nearest_node_to_current = -1
              min_dist_to_current = float('inf')
              for node in temp_unvisited:
                  dist = distance_matrix[current_heuristic_node, node]
                  if dist < min_dist_to_current:
                      min_dist_to_current = dist
                      nearest_node_to_current = node

              if nearest_node_to_current != -1:
                  estimated_remaining_cost += min_dist_to_current
                  current_heuristic_node = nearest_node_to_current
                  temp_unvisited.remove(nearest_node_to_current)
              else:
                  # This case should ideally not happen if temp_unvisited is not empty
                  break

          # Add the cost from the last visited heuristic node to the destination
          estimated_remaining_cost += distance_matrix[current_heuristic_node, destination_node]
      else:
          # If no other nodes are unvisited, the remaining cost is just to go to the destination
          estimated_remaining_cost = distance_matrix[candidate_node, destination_node]

      # The total score is the immediate cost plus the estimated remaining tour cost
      scores[candidate_node] = immediate_cost + estimated_remaining_cost

  # Select the node with the minimum score
  next_node = min(scores, key=scores.get)
  return next_node