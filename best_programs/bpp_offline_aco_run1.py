def priority_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
  """Improved version of `priority_v1`.

  This version further refines the heuristic by:
  1.  **Increasing preference for very tight fits**: The `tightness_factor`
      is now raised to the power of 5: `(1.0 / (remaining_space + 1)^5)`. This
      makes the heuristic even more sensitive to very small remaining spaces,
      strongly boosting pairs that nearly fill a bin.
  2.  **Boosting the filling ratio importance**: The `filling_ratio_factor`
      is now raised to the power of 4: `(sum_demands / capacity)^4`. This
      gives an even stronger non-linear preference to pairs that utilize a
      significant portion of the bin's capacity.
  3.  **Strengthening the relative size difference penalty**: The
      `difference_penalty_factor` is changed to `1.0 / (1 + (abs_diff_demands / capacity)^2)`.
      By squaring the relative difference in the denominator, this factor applies
      a more aggressive penalty for larger discrepancies in item sizes,
      further encouraging more homogeneously-sized pairs.

  The full heuristic score for a feasible pair [i, j] is calculated as:
  (1.0 / (remaining_space + 1)^5) * (sum_demands / capacity)^4 * (1.0 / (1 + (abs_diff_demands / capacity)^2))

  Infeasible pairs (sum_demands > capacity) receive a score of 0, and an item
  cannot be paired with itself (diagonal elements are set to 0).

  Args:
      demand : np.ndarray
          1D array of demand values for n items (shape: (n,)).
      capacity : int
          Integer specifying the maximum capacity of each bin.

  Returns:
      np.ndarray
          2D heuristic matrix of shape (n, n), representing the improved
          priorities for grouping items.
  """
  n = demand.shape[0]

  # Calculate the sum of demands for all possible pairs [i, j]
  sum_demands = demand[:, None] + demand[None, :]

  # Calculate the remaining space if items i and j are placed together
  remaining_space = capacity - sum_demands

  # Determine which pairs are feasible (sum of demands <= capacity)
  is_feasible = remaining_space >= 0

  # Calculate the 'tightness' factor with an inverse raised to the power of 5 (CHANGE 1)
  tightness_factor = np.where(is_feasible, 1.0 / ((remaining_space + 1)**5), 0)

  # Calculate the 'filling ratio' factor: raised to the power of 4 (CHANGE 2)
  filling_ratio_factor = (sum_demands / capacity)**4 

  # Calculate the absolute difference in demands between items i and j
  abs_diff_demands = np.abs(demand[:, None] - demand[None, :])

  # Calculate the 'difference penalty' factor, with squared relative difference (CHANGE 3)
  # The denominator `1 + (abs_diff_demands / capacity)**2` will always be >= 1,
  # preventing division by zero and ensuring the factor is <= 1.
  difference_penalty_factor = np.where(is_feasible, 1.0 / (1 + (abs_diff_demands / capacity)**2), 0)

  # Combine all three factors for feasible pairs.
  # Infeasible pairs (determined by `is_feasible`) receive a score of 0.
  priority_matrix = np.where(is_feasible, 
                             tightness_factor * filling_ratio_factor * difference_penalty_factor, 
                             0)

  # Set diagonal elements to 0, as an item cannot be paired with itself.
  np.fill_diagonal(priority_matrix, 0)

  return priority_matrix