def priority_v2(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
  """
  An alternative heuristic for CVRP edge selection that prioritizes a balanced
  approach to route construction by considering multiple facets of edge desirability.

  This version aims to strike a sophisticated balance between:
  - **Proximity:** Encouraging short travel segments.
  - **Demand Density:** Prioritizing edges that serve significant demand efficiently.
  - **Capacity Balance:** Ensuring routes are built to allow for subsequent stops.

  Key components of this heuristic include:

  1.  **Inverse Distance:** `1.0 / (distance + epsilon)`
      A fundamental factor to encourage shorter travel distances, a primary objective in routing problems.

  2.  **Demand-to-Distance Ratio:** `demands[j] / (distance + epsilon)`
      This term prioritizes customers that offer a high amount of demand relative to the
      travel distance to reach them. This aims to make "productive" trips.

  3.  **Capacity Utilization Efficiency:** `(capacity - demands[j] + epsilon) / (demands[j] + epsilon)`
      This factor evaluates how efficiently capacity is utilized. A higher score indicates that
      the customer's demand is a reasonable fraction of the remaining capacity, leaving room for
      more stops or that the demand is small relative to capacity. It penalizes edges where
      demand is very low or very high compared to the available capacity.

  4.  **Route Completion Potential:** `(capacity - demands[j]) / (distance + epsilon)`
      This component assesses the efficiency of extending a route by considering how much
      capacity is preserved relative to the travel cost. It rewards edges that consume
      less capacity per unit of distance traveled.

  5.  **Customer Value per Distance:** `(1.0 + demands[j]) / (distance + epsilon)`
      This metric quantifies the "value" of traversing an edge by considering the demand of the customer
      and the travel cost. It suggests that edges serving higher demand customers are more desirable,
      especially if the travel distance is low.

  The heuristic combines these factors multiplicatively. The intention is to favor edges that:
  - Are short (inverse distance)
  - Offer high demand with minimal travel cost (demand-to-distance ratio, customer value per distance)
  - Lead to balanced capacity utilization for potential future stops (capacity utilization efficiency, route completion potential).

  This comprehensive approach aims to guide the ACO towards constructing high-quality,
  capacitated vehicle routes by considering both immediate travel efficiency and the
  strategic implications for the overall route structure.

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
  heuristic_matrix = np.zeros((num_nodes, num_nodes))
  demands = np.asarray(demands)

  epsilon = 1e-9  # Small value for numerical stability.

  for i in range(num_nodes):
    for j in range(num_nodes):
      if i == j:
        heuristic_matrix[i, j] = 0.0  # Cannot move to the same node.
        continue

      distance = distance_matrix[i, j]
      demand_j = demands[j]

      # Moves to the depot (node 0) have zero heuristic value for service.
      if j == 0:
        heuristic_matrix[i, j] = 0.0
        continue

      # If demand is negligible or capacity is zero, no heuristic value.
      if demand_j < epsilon or capacity < epsilon:
        heuristic_matrix[i, j] = 0.0
        continue

      # If demand for customer j exceeds vehicle capacity, this edge is invalid.
      if demand_j > capacity:
        heuristic_matrix[i, j] = 0.0
        continue

      # Handle zero distance: highest priority for valid customer moves.
      if distance < epsilon:
        if demand_j <= capacity:
          heuristic_matrix[i, j] = 1e9  # High priority for zero-distance travel to valid customer
        else:
          heuristic_matrix[i, j] = 0.0  # Cannot serve if demand exceeds capacity
        continue

      # Calculate the components of the heuristic.

      # 1. Inverse Distance: Prioritize shorter edges.
      inv_distance = 1.0 / (distance + epsilon)

      # 2. Demand-to-Distance Ratio: High demand relative to travel distance.
      demand_to_distance_ratio = demand_j / (distance + epsilon)

      # 3. Capacity Utilization Efficiency: Balance of remaining capacity vs. demand.
      #    Higher score means demand is a good fraction of capacity, leaving room.
      capacity_utilization_efficiency = (capacity - demand_j + epsilon) / (demands[j] + epsilon)

      # 4. Route Completion Potential: Capacity left per unit of distance.
      route_completion_potential = (capacity - demand_j) / (distance + epsilon)

      # 5. Customer Value per Distance: Demand served per unit of distance.
      customer_value_per_distance = (1.0 + demands[j]) / (distance + epsilon)

      # Combine the components multiplicatively.
      heuristic_value = inv_distance * \
                        demand_to_distance_ratio * \
                        capacity_utilization_efficiency * \
                        route_completion_potential * \
                        customer_value_per_distance

      heuristic_matrix[i, j] = heuristic_value

  # Ensure non-negative and finite values.
  heuristic_matrix = np.maximum(0, heuristic_matrix)
  heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=0.0, posinf=1e9, neginf=0.0)

  return heuristic_matrix