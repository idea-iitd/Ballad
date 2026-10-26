def priority_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
  """
  A heuristic that prioritizes pairs of items that are individually large,
  and whose combined demand fills the bin as much as possible without exceeding
  capacity. It aims to pair large items together if they create a very efficient
  fit within a bin.

  The heuristic score for a pair (i, j) is calculated as follows:
  1.  **Feasibility Check:** Pairs where `demand[i] + demand[j] > capacity`
      are assigned a score of 0.
  2.  **Individual Item Importance:** The score is proportional to the
      `minimum` of the demands of the two items. This encourages pairing
      two large items together.
  3.  **Tightness of Fit:** The score is inversely proportional to the
      *normalized remaining capacity*. This means pairs that leave very little
      space are highly prioritized. The normalized remaining capacity is
      `1.0 - (demand[i] + demand[j]) / capacity`. The inverse is
      `1.0 / (1.0 - (demand[i] + demand[j]) / capacity + epsilon)`.

  The overall score for a feasible pair (i, j) is:
  `min(demand[i], demand[j]) * (1.0 / (1.0 - (demand[i] + demand[j]) / capacity + epsilon))`

  This heuristic aims to balance:
  -   **Pairing of large items:** `min(demand[i], demand[j])` encourages pairing two significant items.
  -   **Maximizing bin utilization (tight fit):** The inverse of normalized remaining capacity strongly rewards pairs that fill the bin almost completely.

  Args:
      demand : np.ndarray
          1D array of demand values for n items (shape: (n,)).
      capacity : int
          Integer specifying the maximum capacity of each bin.

  Returns:
      np.ndarray
          2D heuristic matrix of shape (n, n).
          Entry [i, j] signifies the prior indicator of how suitable it is
          to assign item i and item j to the same bin.
  """
  n = demand.shape[0]
  if n == 0:
      return np.zeros((0, 0))

  # Calculate the sum of demands for all possible pairs (i, j)
  demand_col = demand[:, np.newaxis]
  sum_demands_matrix = demand_col + demand

  # Initialize priority matrix with zeros
  priority_matrix = np.zeros((n, n), dtype=float)

  # Create a mask for feasible pairs (where sum of demands <= capacity)
  feasible_mask = sum_demands_matrix <= capacity

  # Handle the case where all demands are zero
  if np.all(demand == 0):
      return np.zeros((n, n))

  # Calculate the minimum demand of the two items in a pair
  min_individual_demand = np.minimum(demand_col, demand)

  # Add a small epsilon to avoid division by zero
  epsilon = 1e-9

  # Calculate the normalized remaining capacity for feasible pairs.
  # Normalized remaining capacity = 1 - (sum_demands_matrix / capacity)
  normalized_remaining_capacity = 1.0 - (sum_demands_matrix[feasible_mask] / capacity)

  # Calculate the inverse of the normalized remaining capacity for feasible pairs.
  # This term becomes very large when normalized_remaining_capacity is close to 0 (tight fit).
  inverse_normalized_remaining_capacity_values = 1.0 / (normalized_remaining_capacity + epsilon)

  # Combine the factors for feasible pairs:
  # - min_individual_demand[feasible_mask]: Prioritize pairs where both items are relatively large.
  # - inverse_normalized_remaining_capacity_values: Strongly prioritize pairs that
  #   leave very little normalized wasted space (tight fit).
  priority_matrix[feasible_mask] = (
      min_individual_demand[feasible_mask]
      * inverse_normalized_remaining_capacity_values
  )

  # Set diagonal elements to 0, as an item cannot be paired with itself
  np.fill_diagonal(priority_matrix, 0.0)

  return priority_matrix