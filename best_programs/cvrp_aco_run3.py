def priority_v2(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
  """
  An advanced heuristic combining multiple factors with dynamic adjustments.

  This heuristic focuses on:
  1.  **Distance:** Prioritizes shorter edges.
  2.  **Demand-Capacity Ratio:** Balances the importance of high-demand customers
      with the risk of exceeding capacity.
  3.  **Customer Density Around 'j':** Favors nodes `j` that are closer to other
      unserved customers, encouraging efficient clustering.
  4.  **Route Completion Likelihood:** Estimates the probability of successfully
      serving remaining customers given the current vehicle's remaining capacity.
  5.  **Proximity to Depot (for return trip):** Considers the distance back to
      the depot, especially crucial when capacity is low or many customers are
      served.
  6.  **Edge Value (i, j) vs. Avg. Customer Distance:** Dynamically adjusts
      the importance of the current edge based on how it compares to the average
      reach to other customers from node `i`.

  Args:
      distance_matrix : np.ndarray
          2D array of distances between all node pairs (shape: n by n).
      coordinates : np.ndarray
          2D array of Euclidean coordinates of nodes (shape: n by 2).
      demands : list or np.ndarray
          Vector of customer demand requirements (shape: n). The depot node
          (index 0) has demand = 0.
      capacity : int or float
          Integer capacity constraint representing the maximum capacity
          of each vehicle.

  Returns:
      np.ndarray
          Heuristic matrix of the same shape as the distance_matrix, where
          entry [i, j] represents prior indicators of how promising it is
          to move from node i to node j.
  """
  n = distance_matrix.shape[0]
  heuristic_matrix = np.zeros_like(distance_matrix)
  demands = np.asarray(demands)
  epsilon = 1e-6

  # --- Pre-calculations ---
  customer_indices = np.where(demands > 0)[0]
  num_customers = len(customer_indices)

  # Average customer demand for normalization.
  total_demand_customers = np.sum(demands[demands > 0])
  avg_customer_demand = total_demand_customers / num_customers if num_customers > 0 else 0

  # Average distance from each node to all other *customer* nodes.
  avg_dist_to_customers = np.zeros(n)
  for i in range(n):
      if num_customers > 0:
          distances_from_i_to_customers = distance_matrix[i, customer_indices]
          # Exclude self-distance if node i is a customer itself.
          if i in customer_indices:
              self_dist_idx = np.where(customer_indices == i)[0][0]
              distances_from_i_to_customers = np.delete(distances_from_i_to_customers, self_dist_idx)
          if len(distances_from_i_to_customers) > 0:
              avg_dist_to_customers[i] = np.mean(distances_from_i_to_customers)

  # Distance from each node to the depot (node 0).
  dist_to_depot = distance_matrix[:, 0]

  # --- Heuristic Factor Weights (Tunable Parameters) ---
  w_dist = 2.5               # Weight for inverse distance
  w_demand_cap_ratio = 1.5   # Weight for demand relative to capacity
  w_customer_density_j = 1.0 # Weight for how close j is to other customers
  w_route_completion = 1.2   # Weight for likelihood of completing a route
  w_return_trip = 0.8        # Weight for proximity to depot for return trip
  w_edge_vs_avg_reach = 0.9  # Dynamic adjustment based on edge vs. avg reach

  for i in range(n):
    for j in range(n):
      if i == j:
        heuristic_matrix[i, j] = 0
        continue

      distance_ij = distance_matrix[i, j]
      demand_j = demands[j]

      # --- Infeasibility Check ---
      if demand_j > capacity:
        heuristic_matrix[i, j] = epsilon
        continue

      # --- Factor 1: Distance ---
      # Shorter distances are preferred.
      distance_score = (1.0 / (distance_ij + epsilon))**w_dist

      # --- Factor 2: Demand-Capacity Ratio ---
      # Prioritize customers with higher demand relative to capacity, but not so high
      # that they are infeasible or leave no room for other customers.
      # Using a sigmoid-like function to dampen extreme values and favor moderate ones.
      demand_cap_ratio = demand_j / capacity
      demand_cap_score = (1 - np.exp(-5 * demand_cap_ratio))**w_demand_cap_ratio if demand_cap_ratio > epsilon else epsilon

      # --- Factor 3: Customer Density Around 'j' ---
      # How well-connected is node `j` to other potential customers?
      # A smaller average distance to other customers is better.
      customer_density_score = 1.0
      if avg_dist_to_customers[j] > epsilon and distance_ij > epsilon:
          # Inverse ratio: smaller avg_dist_to_customers[j] is better.
          # Normalize by distance_ij to favor edges that lead into dense areas.
          customer_density_score = (avg_dist_to_customers[j] / distance_ij)**w_customer_density_j
      elif avg_dist_to_customers[j] == 0 and num_customers > 1:
          customer_density_score = 1.0 # Neutral if j is isolated but there are other customers
      else:
          customer_density_score = epsilon**w_customer_density_j

      # --- Factor 4: Route Completion Likelihood ---
      # Estimate the probability of completing a route from `j` given remaining capacity.
      # This is approximated by the ratio of remaining capacity to the average demand
      # of all customers.
      remaining_capacity_after_j = capacity - demand_j
      route_completion_score = 1.0
      if avg_customer_demand > epsilon and remaining_capacity_after_j > epsilon:
          route_completion_score = (remaining_capacity_after_j / avg_customer_demand)**w_route_completion
      elif avg_customer_demand == 0:
          route_completion_score = 1.0 # If no customers, always feasible to complete
      else:
          route_completion_score = epsilon # No capacity left

      # --- Factor 5: Proximity to Depot (for return trip) ---
      # Especially important when capacity is getting low, or many customers
      # have been served.
      return_trip_score = 1.0
      if dist_to_depot[j] > epsilon and distance_ij > epsilon:
          # Inverse ratio: smaller distance to depot relative to current edge is better.
          return_trip_score = (dist_to_depot[j] / distance_ij)**w_return_trip
      elif dist_to_depot[j] == 0:
          return_trip_score = 1.0 # Already at depot
      else:
          return_trip_score = epsilon**w_return_trip

      # --- Factor 6: Edge Value (i, j) vs. Avg. Customer Distance ---
      # Dynamically adjust heuristic based on how attractive the current edge is
      # compared to the average reach from node `i`.
      edge_vs_avg_reach_score = 1.0
      if avg_dist_to_customers[i] > epsilon and distance_ij > epsilon:
          # If edge_ij is much shorter than average reach from i, it's good.
          edge_vs_avg_reach_score = (avg_dist_to_customers[i] / distance_ij)**w_edge_vs_avg_reach
      else:
          edge_vs_avg_reach_score = 1.0 # Neutral if no clear comparison

      # --- Combine Factors ---
      # Multiplicative combination for synergistic effect.
      heuristic_value = (distance_score *
                         demand_cap_score *
                         customer_density_score *
                         route_completion_score *
                         return_trip_score *
                         edge_vs_avg_reach_score)

      # Ensure depot node (demand 0) has minimal heuristic value.
      if demand_j == 0:
          heuristic_value = epsilon

      heuristic_matrix[i, j] = heuristic_value

  # --- Final Cleanup ---
  # Replace NaNs and infinities, ensure all values are positive and non-zero.
  heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=epsilon, posinf=1e9, neginf=-1e9)
  heuristic_matrix[heuristic_matrix <= 0] = epsilon

  return heuristic_matrix