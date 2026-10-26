def priority_v1(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
  """
  An improved heuristic for CVRP edge selection, building upon `priority_v0`.

  This version aims to refine the heuristic by:
  1.  **More Sophisticated Demand Handling:** Consider not just individual demand,
      but the *cumulative* demand on a potential route segment, relative to capacity.
      This is challenging to do purely statically. We'll approximate by considering
      the demand of the *source* node more strongly and how it contributes to a partial route load.
  2.  **Dynamic Cost Adjustment (Static Approximation):** Make edge costs
      more sensitive to how they fit into a potential route. For example,
      edges that lead to a high concentration of remaining demand might be
      less preferred unless they are close to the depot.
  3.  **Customer "Reachability" Score:** Nodes that are "harder" to reach (further
      from clusters, higher demand) might be prioritized earlier in a route
      to ensure they are not missed.
  4.  **Balancing Inter-vehicle Load:** While a static heuristic can't directly
      balance loads between vehicles, it can encourage choices that *tend* to
      lead to more balanced routes (e.g., by favoring shorter routes or
      routes that visit nodes with intermediate demands).
  5.  **Refined Peripheralness/Centrality:** Differentiate between peripheralness
      for the source and destination nodes.

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
          to move from node i to node j, considering distance, demand,
          and capacity constraints.
  """
  num_nodes = distance_matrix.shape[0]
  safe_distance_matrix = np.where(distance_matrix == 0, 1e-9, distance_matrix)
  demands = np.asarray(demands)

  if capacity <= 0:
    return np.zeros_like(distance_matrix)

  # --- Factor 1: Inverse Distance ---
  # Always prefer shorter edges.
  inverse_distance = 1.0 / safe_distance_matrix

  # --- Factor 2: Destination Demand Considerations ---
  # Higher demand at destination is less preferred for the *current* route segment,
  # as it consumes capacity. But essential customers must be visited.
  # We penalize high demand destinations. A good approach is to consider demand relative to capacity.
  # Using 1 / (1 + (demand/capacity)^2) makes higher demands much less attractive.
  demand_destination_score = 1.0 / (1.0 + (demands / capacity)**2)
  demand_destination_matrix = np.tile(demand_destination_score, (num_nodes, 1))

  # --- Factor 3: Source Node Demand and Load Consideration ---
  # If node `i` has high demand *and* it's likely to be part of a route already,
  # departing from it might be less ideal if it leads to exceeding capacity.
  # This is a static approximation. We'll favor departing from nodes with lower *remaining* demand.
  # This is subtly different from v0: we want to penalize nodes that *already have high demand fulfilled*.
  # For a static heuristic, we'll use the current demand as a proxy for load.
  demand_source_score = np.zeros(num_nodes)
  # Prefer departing from nodes with lower demand.
  demand_source_score[1:] = 1.0 / (1.0 + demands[1:] / capacity)
  demand_source_score[0] = 1.0 # Depot is always a good source to start from.
  demand_source_matrix = np.tile(demand_source_score, (num_nodes, 1))

  # --- Factor 4: Peripheralness of Destination ---
  # This is about where a node is located geographically.
  # Peripheral nodes might be better visited towards the end of routes.
  # We'll use average distance to all other nodes as a proxy for peripheralness.
  avg_distances_from_node = np.mean(safe_distance_matrix, axis=1)
  max_avg_distance = np.max(avg_distances_from_node)
  # Higher score means more peripheral.
  peripheralness_score = (avg_distances_from_node / (max_avg_distance + 1e-9)) if max_avg_distance > 1e-9 else np.zeros(num_nodes)
  peripheralness_score[0] = 0.0 # Depot is not peripheral.
  peripheralness_destination_matrix = np.tile(peripheralness_score, (num_nodes, 1))

  # --- Factor 5: Centrality of Source ---
  # Conversely, starting from a more central node might be beneficial for initiating routes.
  # Let's define centrality as the inverse of peripheralness (or average distance to nearest neighbors).
  # A node close to many other nodes is central.
  # For simplicity, we'll use the inverse of peripheralness (but handle division by zero).
  centrality_score = np.zeros(num_nodes)
  # Centrality is higher for nodes with lower peripheralness score.
  centrality_score[1:] = 1.0 / (peripheralness_score[1:] + 1e-9)
  centrality_score[0] = 1.0 # Depot can be considered a highly central reference point.
  # Normalize centrality for consistent scaling.
  max_centrality = np.max(centrality_score)
  normalized_centrality = (centrality_score / (max_centrality + 1e-9)) if max_centrality > 1e-9 else np.ones(num_nodes)
  centrality_source_matrix = np.tile(normalized_centrality, (num_nodes, 1))

  # --- Factor 6: Depot Return Incentive ---
  # Strongly encourage returning to the depot from nodes that are far away or have significant demand.
  distance_to_depot = safe_distance_matrix[:, 0]
  max_dist_to_depot = np.max(distance_to_depot[distance_to_depot > 1e-9]) if np.max(distance_to_depot[distance_to_depot > 1e-9]) > 1e-9 else 1.0
  normalized_dist_to_depot = distance_to_depot / max_dist_to_depot
  normalized_demand = demands / capacity

  # Combine distance and demand to create a return incentive. Higher values mean stronger incentive.
  depot_return_incentive_from_node = normalized_dist_to_depot * 1.2 + normalized_demand * 1.8
  depot_return_incentive_from_node = np.clip(depot_return_incentive_from_node, 0.0, 5.0) # Cap at a higher value

  depot_return_matrix = np.tile(depot_return_incentive_from_node, (num_nodes, 1))

  # --- Factor 7: Customer "Reachability" Score (Approximation) ---
  # This is to ensure that "difficult" customers (e.g., far away, high demand) are not
  # left until the very end in a way that makes routes infeasible or inefficient.
  # Higher score means "harder to reach" or more critical to visit.
  reachability_score = np.zeros(num_nodes)
  # Combine normalized distance from depot and normalized demand.
  reachability_score[1:] = normalized_dist_to_depot[1:] + (demands[1:] / capacity)
  # Normalize reachability score
  max_reachability = np.max(reachability_score) if np.max(reachability_score) > 1e-9 else 1.0
  normalized_reachability = reachability_score / max_reachability
  normalized_reachability[0] = 0.0 # Depot is trivially reachable.

  # This score is applied to the destination node `j` of an edge `(i, j)`.
  reachability_destination_matrix = np.tile(normalized_reachability, (num_nodes, 1))


  # --- Combine factors ---
  # Adjusted weights for v1, emphasizing distance, demand handling, and reachability.
  # These weights are crucial for tuning the heuristic's behavior.
  weight_distance = 4.0              # Strong preference for shorter edges.
  weight_demand_dest = 2.5           # Penalize high demand destinations significantly.
  weight_demand_src = 1.2            # Slightly favor departing from low-demand sources.
  weight_peripheralness_dest = 1.5   # Favor peripheral destinations (for later in route).
  weight_centrality_src = 1.0        # Favor central sources (for earlier in route).
  weight_depot_return = 1.5          # Strong incentive to return to depot.
  weight_reachability = 1.8          # Prioritize "harder to reach" customers.

  # The heuristic is a product of powered factors.
  heuristic_matrix = (
      np.power(inverse_distance, weight_distance)
      * np.power(demand_destination_matrix, weight_demand_dest)
      * np.power(demand_source_matrix, weight_demand_src)
      * np.power(peripheralness_destination_matrix, weight_peripheralness_dest)
      * np.power(centrality_source_matrix, weight_centrality_src)
      * np.power(depot_return_matrix, weight_depot_return)
      * np.power(reachability_destination_matrix, weight_reachability)
  )

  # --- Post-processing ---

  # 1. Hard constraint: If a customer's demand exceeds vehicle capacity, no edge involving that customer is valid.
  # This is a critical CVRP constraint that cannot be purely handled by heuristics.
  invalid_demand_mask = demands > capacity
  # Mark edges where the destination node has demand exceeding capacity as invalid (heuristic = 0)
  heuristic_matrix[:, invalid_demand_mask] = 0.0
  # Mark edges where the source node has demand exceeding capacity as invalid.
  heuristic_matrix[invalid_demand_mask, :] = 0.0

  # 2. Ensure a minimum heuristic value for exploration.
  # Any edge that hasn't been explicitly set to 0 should have a small positive value.
  min_heuristic_value = 1e-6
  heuristic_matrix[heuristic_matrix < min_heuristic_value] = min_heuristic_value

  # Handle potential NaNs or Infs from calculations
  heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=min_heuristic_value, posinf=1e9, neginf=-1e9)

  return heuristic_matrix