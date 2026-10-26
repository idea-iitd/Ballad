def priority_v2(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
  """
  Selects the next node to visit in the Travelling Salesman Problem (TSP) tour,
  prioritizing nodes that lead to a shorter estimated remaining tour, with a
  bias towards the immediate nearest neighbor when few nodes remain.

  This version is an improvement on `priority_v1` by more explicitly
  incorporating the total tour cost estimation into the priority calculation.
  It also uses a simpler, more robust dynamic weighting scheme.

  The logic:
  1. If only one node is unvisited, return it.
  2. For each `potential_next_node` in `unvisited_nodes`:
     a. Calculate the direct cost from `current_node` to `potential_next_node` (immediate cost).
     b. Estimate the "future path cost" from `potential_next_node` to the `destination_node`.
        This estimation is done by constructing a *temporary* greedy tour from
        `potential_next_node` through all other `unvisited_nodes - {potential_next_node}`,
        and finally back to `destination_node`.
     c. Dynamically determine weights for immediate and future costs:
        - As the number of unvisited nodes decreases (i.e., the tour progresses),
          the `w_immediate_cost` increases, and `w_future_path_cost` decreases.
        - This aims to make the heuristic behave more like a simple Nearest Neighbor
          at the end of the tour, and more like a lookahead-based heuristic at the beginning.
     d. Calculate the priority score as a weighted sum of immediate and future costs.
     e. Select the `potential_next_node` with the minimum priority score.

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
          according to the heuristic used in this function.
  """
  num_unvisited = len(unvisited_nodes)
  total_nodes = distance_matrix.shape[0]

  if num_unvisited == 1:
    return list(unvisited_nodes)[0]

  best_next_node = -1
  min_priority_score = float('inf')

  # --- Dynamic Weighting Strategy ---
  # The goal is to shift from a broader lookahead to a more myopic (nearest neighbor)
  # focus as the tour progresses.
  #
  # Let's define a "progress" metric. A simple way is to consider the fraction of
  # nodes remaining to be visited *after* the next step.
  # If num_unvisited = N-1 (start), progress is near 0.
  # If num_unvisited = 1 (end), progress is near 1.
  #
  # We can use this progress to adjust weights.
  # `progress_factor` will range from ~0 (at start) to ~1 (at end).
  # Let's use `(total_nodes - 1 - num_unvisited) / (total_nodes - 1)` as the progress factor.
  # However, if `total_nodes` is 1 or 2, the denominator can be zero.
  # A simpler and more robust way: use `num_unvisited` directly.
  #
  # When `num_unvisited` is large, `w_future` should be higher and `w_immediate` lower.
  # When `num_unvisited` is small, `w_immediate` should be higher and `w_future` lower.

  # Define base weights
  base_immediate_weight = 1.0
  base_future_weight = 0.8

  # Calculate an adjustment factor based on the number of unvisited nodes.
  # This factor will be higher when `num_unvisited` is small.
  # Let's scale `num_unvisited` relative to `total_nodes`.
  # A factor from 0 to 1, where 1 means `num_unvisited` is 1, and 0 means `num_unvisited` is `total_nodes - 1`.
  # This makes `adjustment` closer to 1 at the end of the tour, and closer to 0 at the start.

  if total_nodes <= 2: # No dynamic adjustment needed for trivial cases
      adjustment = 0.0
  else:
      # `num_unvisited` goes from `total_nodes - 1` down to `1`.
      # We want `adjustment` to be close to 1 when `num_unvisited` is 1.
      # We want `adjustment` to be close to 0 when `num_unvisited` is `total_nodes - 1`.
      # Linear mapping: `adjustment = (num_unvisited - 1) / (total_nodes - 2)`
      adjustment = (num_unvisited - 1) / (total_nodes - 2)

  # Adjust weights:
  # As `adjustment` increases (tour progresses), increase `w_immediate` and decrease `w_future`.
  w_immediate_cost = base_immediate_weight + adjustment * 0.8  # e.g., increases from 1.0 to 1.8
  w_future_path_cost = base_future_weight * (1 - adjustment * 0.6) # e.g., decreases from 0.8 to 0.8 * (1-0.6) = 0.32

  for potential_next_node in unvisited_nodes:
    # a. Calculate the direct cost from `current_node` to `potential_next_node`.
    immediate_cost = distance_matrix[current_node, potential_next_node]

    # b. Estimate the "future path cost" from `potential_next_node` to the `destination_node`.
    # This is a full greedy Nearest Neighbor tour starting from `potential_next_node`.
    future_path_cost = 0

    remaining_unvisited_for_future = unvisited_nodes - {potential_next_node}

    if not remaining_unvisited_for_future:
        # If this is the last unvisited node, the future path is just to the destination.
        future_path_cost = distance_matrix[potential_next_node, destination_node]
    else:
        # Perform a greedy Nearest Neighbor tour from `potential_next_node` through all remaining nodes.
        temp_current_for_future = potential_next_node
        temp_unvisited_for_future = copy(remaining_unvisited_for_future)

        while temp_unvisited_for_future:
            closest_remaining_node_in_future = -1
            min_dist_to_remaining_in_future = float('inf')

            for node in temp_unvisited_for_future:
                dist = distance_matrix[temp_current_for_future, node]
                if dist < min_dist_to_remaining_in_future:
                    min_dist_to_remaining_in_future = dist
                    closest_remaining_node_in_future = node

            future_path_cost += min_dist_to_remaining_in_future
            temp_current_for_future = closest_remaining_node_in_future
            temp_unvisited_for_future.remove(closest_remaining_node_in_future)

        # Add the distance from the last visited node in the greedy path to the destination.
        future_path_cost += distance_matrix[temp_current_for_future, destination_node]

    # c. Calculate the priority score using dynamic weights.
    priority_score = (w_immediate_cost * immediate_cost) + \
                     (w_future_path_cost * future_path_cost)

    # d. Select the `potential_next_node` with the minimum priority score.
    if priority_score < min_priority_score:
      min_priority_score = priority_score
      best_next_node = potential_next_node

  return best_next_node