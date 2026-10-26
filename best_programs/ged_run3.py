def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    This version of the priority function aims to further refine the node mapping probabilities
    by considering more complex neighborhood structures and edge attributes.

    It incorporates:
    1.  **Node Degree Profile Similarity:** Measures similarity based on the degrees of neighbors.
    2.  **Common Neighbors Weighting:** Emphasizes common neighbors that have high initial mapping probabilities.
    3.  **Edge Consistency:** Penalizes mappings where an edge exists in one graph but not the corresponding "edge" in the mapping of the other graph.
    4.  **Node Attribute Similarity (if applicable):** While not explicitly provided in the function signature, this is a placeholder for future extensions where node labels/attributes would be considered.

    Args:
        graph1: Adjacency matrix of the first graph.
        graph2: Adjacency matrix of the second graph.
        weights: Initial weight matrix (n1 x n2) representing prior mapping probabilities.

    Returns:
        A refined weight matrix (n1 x n2) with probabilities for mapping nodes.
  """
  n1 = len(graph1)
  n2 = len(graph2)

  # Ensure weights matrix has the correct dimensions
  if len(weights) != n1 or (n1 > 0 and len(weights[0]) != n2):
      raise ValueError(f"Input weights matrix dimensions ({len(weights)}x{len(weights[0]) if n1 > 0 else 0}) do not match graph dimensions ({n1}x{n2})")

  refined_weights = [[float(weights[i][j]) for j in range(n2)] for i in range(n1)]

  # --- Feature Engineering ---

  # Feature 1: Node Degree Profile Similarity
  # For node `i` in G1, consider the degrees of its neighbors.
  # For node `j` in G2, consider the degrees of its neighbors.
  # This feature captures structural similarity beyond simple node degrees.
  degree_profile_similarities = np.zeros((n1, n2))
  for i in range(n1):
    neighbor_degrees_g1 = [sum(graph1[k]) for k in range(n1) if graph1[i][k] == 1]
    for j in range(n2):
      neighbor_degrees_g2 = [sum(graph2[l]) for l in range(n2) if graph2[j][l] == 1]

      if not neighbor_degrees_g1 and not neighbor_degrees_g2:
        degree_profile_similarities[i, j] = 1.0
      elif not neighbor_degrees_g1 or not neighbor_degrees_g2:
        degree_profile_similarities[i, j] = 0.0
      else:
        # Use a simple similarity measure (e.g., based on sorted degree lists)
        sorted_degrees_g1 = sorted(neighbor_degrees_g1)
        sorted_degrees_g2 = sorted(neighbor_degrees_g2)
        min_len = min(len(sorted_degrees_g1), len(sorted_degrees_g2))
        sim_score = 0.0
        for k in range(min_len):
          sim_score += 1.0 - abs(sorted_degrees_g1[k] - sorted_degrees_g2[k]) / max(sorted_degrees_g1[k], sorted_degrees_g2[k], 1e-6)
        if min_len > 0:
            degree_profile_similarities[i, j] = sim_score / min_len
        else:
            degree_profile_similarities[i, j] = 0.0

  # Feature 2: Weighted Common Neighbors
  # Common neighbors where the mapping between them has a high initial weight.
  weighted_common_neighbors = np.zeros((n1, n2))
  for i in range(n1):
    for j in range(n2):
      score = 0.0
      for k in range(n1):
        if graph1[i][k] == 1: # k is a neighbor of i
          for l in range(n2):
            if graph2[j][l] == 1 and graph1[i][l] == 1 and weights[k][l] > 0.7: # l is a neighbor of j, and k maps to l with high probability
               score += weights[k][l]
      weighted_common_neighbors[i, j] = score

  # Feature 3: Edge Consistency (Simplified)
  # Check if the presence of an edge in G1 between i and k is "consistent"
  # with the presence of an edge in G2 between j and l, where i maps to j and k maps to l.
  # This is a weaker form of edge consistency, focusing on direct neighborhood mappings.
  edge_consistency_score = np.zeros((n1, n2))
  for i in range(n1):
    for j in range(n2):
      for k in range(n1):
        if graph1[i][k] == 1: # Edge (i, k) exists in G1
          for l in range(n2):
            if graph2[j][l] == 1 and weights[k][l] > 0.5: # Edge (j, l) exists in G2 and k maps to l with moderate probability
              # If edge (i,k) exists and we map i->j and k->l, we expect edge (j,l) to exist.
              # This is more of a constraint satisfaction check.
              if graph2[j][l] == 1:
                edge_consistency_score[i, j] += weights[k][l] * 1.0 # Reward for consistency
              else:
                edge_consistency_score[i, j] -= weights[k][l] * 0.5 # Penalize for inconsistency

  # --- Combining Features ---
  # Tunable weights for different features.
  feature_weights = {
      "initial": 1.0,
      "degree_profile": 0.4,
      "weighted_common_neighbors": 0.3,
      "edge_consistency": 0.2,
      "degree_similarity": 0.5, # Re-include from v1 for robustness
      "neighborhood_similarity": 0.3, # Re-include from v1
  }

  # Recompute features from v1 to combine with new ones
  degree_similarities = np.zeros((n1, n2))
  for i in range(n1):
    for j in range(n2):
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])
      max_degree = max(degree1, degree2, 1e-6)
      degree_similarities[i, j] = 1.0 - abs(degree1 - degree2) / max_degree

  neighborhood_similarities = np.zeros((n1, n2))
  for i in range(n1):
    for j in range(n2):
      score = 0.0
      for k in range(n1):
        if graph1[i][k] == 1:
          for l in range(n2):
            if graph2[j][l] == 1:
              score += weights[k][l]
      neighborhood_similarities[i, j] = score

  for i in range(n1):
    for j in range(n2):
      # The `refined_weights` matrix is initialized with `weights`
      refined_weights[i][j] *= feature_weights["initial"]
      refined_weights[i][j] += degree_profile_similarities[i, j] * feature_weights["degree_profile"]
      refined_weights[i][j] += weighted_common_neighbors[i, j] * feature_weights["weighted_common_neighbors"]
      refined_weights[i][j] += edge_consistency_score[i, j] * feature_weights["edge_consistency"]
      refined_weights[i][j] += degree_similarities[i, j] * feature_weights["degree_similarity"]
      refined_weights[i][j] += neighborhood_similarities[i, j] * feature_weights["neighborhood_similarity"]


  # --- Normalization ---
  # Normalize rows to sum to 1.
  for i in range(n1):
    row_sum = sum(refined_weights[i])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum
    else:
      # If a row sums to zero, distribute uniformly.
      if n2 > 0:
        for j in range(n2):
          refined_weights[i][j] = 1.0 / n2

  # --- Post-processing ---
  # Ensure weights are within [0, 1] and clean up any numerical issues.
  for i in range(n1):
    for j in range(n2):
      refined_weights[i][j] = max(0.0, min(1.0, refined_weights[i][j]))

  return refined_weights