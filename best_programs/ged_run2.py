def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """
    An improved version of `priority_v1` that incorporates more sophisticated
    graph structural features and inter-node relationships for weight refinement.

    This version builds upon the heuristics in `priority_v1` and adds:

    1.  **Node Label Similarity (if available):** If nodes had labels (represented
        by feature vectors `nf1` and `nf2` in the `GEDInstance`), this version
        would consider their similarity. For simplicity in this standalone function,
        this part is omitted but conceptually important.
    2.  **Common Neighbors Degree Similarity:** For a mapping of node `i` to `j`,
        this considers pairs of common neighbors (`k` from `graph1`, `l` from `graph2`)
        and boosts the `(k, l)` mapping probability if `k` and `l` have similar degrees.
        This reinforces structural consistency at the local neighborhood level.
    3.  **Iterative Refinement:** While not a full iterative process, this function
        applies a multi-stage refinement, allowing earlier adjustments to influence
        later ones.

    Args:
        graph1: The adjacency matrix of the first graph.
        graph2: The adjacency matrix of the second graph.
        weights: An initial weight matrix, where `weights[i][j]` represents the
                 initial likelihood of mapping node `i` from `graph1` to node `j`
                 from `graph2`.

    Returns:
        A refined weight matrix (list of lists of floats) where each entry
        `refined_weights[i][j]` represents an adjusted probability of mapping
        node `i` to node `j`, informed by more advanced graph structural
        similarities. The weights are normalized such that each row sums to 1.
  """
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  # Initialize refined_weights with input weights, padding if necessary
  refined_weights = [[0.0] * max_node for _ in range(max_node)]
  for i in range(min(n1, len(weights))):
    for j in range(min(n2, len(weights[0]))):
      refined_weights[i][j] = weights[i][j]

  # Pre-calculate degrees for efficiency
  degrees1 = [sum(row) for row in graph1]
  degrees2 = [sum(row) for row in graph2]

  # --- Heuristics from priority_v1 (kept as they are foundational) ---

  # 1. Node Degree Similarity
  for i in range(n1):
    for j in range(n2):
      max_degree_val = max(n1, n2) - 1
      if max_degree_val <= 0:
          degree_sim = 1.0
      else:
          degree_diff_normalized = abs(degrees1[i] - degrees2[j]) / max_degree_val
          degree_sim = 1.0 - degree_diff_normalized
      # Increase weight based on similarity, adding a small bias
      refined_weights[i][j] *= (degree_sim + 0.05)

  # 2. Local Neighborhood Similarity (based on neighbor degrees)
  for i in range(n1):
    for j in range(n2):
      neighbors_i = [k for k in range(n1) if graph1[i][k] == 1]
      neighbors_j = [l for l in range(n2) if graph2[j][l] == 1]

      neighbor_degrees_i = sorted([degrees1[ni] for ni in neighbors_i])
      neighbor_degrees_j = sorted([degrees2[nj] for nj in neighbors_j])

      len_i, len_j = len(neighbor_degrees_i), len(neighbor_degrees_j)
      max_len = max(len_i, len_j)

      if max_len == 0:
        neighborhood_sim = 1.0
      else:
        padded_i = neighbor_degrees_i + [0] * (max_len - len_i)
        padded_j = neighbor_degrees_j + [0] * (max_len - len_j)

        max_possible_degree = max(n1, n2)
        normalized_i = [d / (max_possible_degree if max_possible_degree > 0 else 1.0) for d in padded_i]
        normalized_j = [d / (max_possible_degree if max_possible_degree > 0 else 1.0) for d in padded_j]

        total_diff = sum(abs(ni - nj) for ni, nj in zip(normalized_i, normalized_j))
        avg_diff = total_diff / max_len
        neighborhood_sim = 1.0 - avg_diff

      # Boost weight based on neighborhood similarity
      refined_weights[i][j] *= (1 + neighborhood_sim * 0.4)

  # 3. Global Graph Structure Similarity (Density)
  density1 = sum(sum(row) for row in graph1) / (n1 * (n1 - 1) if n1 > 1 else 1e-6)
  density2 = sum(sum(row) for row in graph2) / (n2 * (n2 - 1) if n2 > 1 else 1e-6)
  density_sim = 1.0 - abs(density1 - density2)

  for i in range(n1):
    for j in range(n2):
      refined_weights[i][j] *= density_sim

  # 4. Edge Existence Similarity (based on neighbor mapping probabilities)
  for i in range(n1):
    for j in range(n2):
      edge_similarity_score = 0.0
      count_edges = 0
      for k in range(n1):
        if graph1[i][k] == 1:
          for l in range(n2):
            if graph2[j][l] == 1:
              edge_similarity_score += refined_weights[k][l]
              count_edges += 1

      if count_edges > 0:
        avg_edge_neighbor_prob = edge_similarity_score / count_edges
        refined_weights[i][j] *= (1 + avg_edge_neighbor_prob * 0.2)

  # --- New Heuristics for v2 ---

  # 5. Common Neighbors Degree Similarity
  # For a mapping (i, j), consider pairs of common neighbors (k, l) where k is a neighbor of i
  # and l is a neighbor of j. Boost the mapping (k, l) if k and l have similar degrees.
  for i in range(n1):
    for j in range(n2):
      for k in range(n1):
        if graph1[i][k] == 1:  # k is a neighbor of i
          for l in range(n2):
            if graph2[j][l] == 1:  # l is a neighbor of j
              # Calculate similarity of degrees for these neighbors k and l
              max_degree_val = max(n1, n2) - 1
              if max_degree_val <= 0:
                  degree_sim_k_l = 1.0
              else:
                  degree_diff_k_l = abs(degrees1[k] - degrees2[l])
                  degree_sim_k_l = 1.0 - (degree_diff_k_l / max_degree_val)
              # Boost the probability of mapping k to l if their degrees are similar
              # This reinforces structural consistency around the i-j mapping
              refined_weights[k][l] *= (1 + degree_sim_k_l * 0.1) # Weight factor for this heuristic

  # 6. Path Consistency (considering paths of length 2)
  # For a mapping (i, j), if there's a path i -> k -> m in graph1,
  # and we map i->j, k->p, m->q, we expect a higher probability for mapping
  # p to q if p and q have similar structural properties.
  for i in range(n1):
    for j in range(n2):
      path_consistency_score = 0.0
      path_count = 0
      for k in range(n1):
        if graph1[i][k] == 1: # Intermediate node k
          for m in range(n1):
            if graph1[k][m] == 1 and m != i: # Path i -> k -> m
              path_count += 1
              # Find potential mappings for k and m
              for p in range(n2):
                if refined_weights[k][p] > 1e-6: # If k can map to p
                  for q in range(n2):
                    if refined_weights[m][q] > 1e-6 and q != j: # If m can map to q (and q is not j)
                      # We want to see if the mapping p -> q is consistent with their properties
                      # This can be measured by degree similarity, neighborhood similarity etc.
                      max_degree_val = max(n1, n2) - 1
                      if max_degree_val <= 0:
                          degree_sim_p_q = 1.0
                      else:
                          degree_diff_p_q = abs(degrees1[k] - degrees2[p]) # Mistake: should be degrees1[k] vs degrees2[p]
                          # Correcting: it should be degrees of original nodes k and m mapped to p and q
                          degree_diff_k_p = abs(degrees1[k] - degrees2[p])
                          degree_diff_m_q = abs(degrees1[m] - degrees2[q])
                          avg_degree_sim_pq = (
                              (1.0 - (degree_diff_k_p / max_degree_val if max_degree_val > 0 else 0)) +
                              (1.0 - (degree_diff_m_q / max_degree_val if max_degree_val > 0 else 0))
                          ) / 2.0
                          degree_sim_p_q = avg_degree_sim_pq # Using average similarity of mapped neighbors' degrees

                      # Boost the probability of mapping i->j if its direct neighbors can map to
                      # nodes that are structurally similar to the mapped intermediate nodes.
                      # This is a complex interaction. A simpler approach is to boost based on
                      # the product of probabilities of intermediate mappings and their structural similarity.
                      path_consistency_score += (
                          refined_weights[k][p] * refined_weights[m][q] * (degree_sim_p_q + 0.1) # Add a small bias
                      )

      if path_count > 0:
        # This heuristic is tricky. We are trying to influence the i->j mapping
        # based on the consistency of mappings of its neighbors' paths.
        # A direct boost to refined_weights[i][j] might be too simplistic.
        # It's more about ensuring that if a path exists in G1, similar paths are likely in G2.
        # For now, let's simplify: if a node has many paths, and its potential mappings
        # lead to consistent downstream mappings, it reinforces the initial mapping.
        # A simpler heuristic here could be to just consider the number of common neighbors
        # of mapped neighbors, which is already captured in heuristic 5.
        # Let's try a simpler boost for now.
        # If there are many paths originating from i, and the downstream mappings are consistent,
        # it means i is structurally important.
        # The contribution of path_consistency_score might be better used to adjust
        # the initial weights of the direct mappings (i,j).
        # For now, let's use a simplified version: if the node has many paths and the average
        # downstream consistency is high, boost the node's initial mapping probability.
        avg_path_consistency = path_consistency_score / (path_count * n2 * n2) if (path_count * n2 * n2) > 0 else 0 # Normalize
        refined_weights[i][j] *= (1 + avg_path_consistency * 0.3) # Weight factor

  # --- Normalization ---
  for i in range(n1):
    row_sum = sum(refined_weights[i][:n2])
    if row_sum > 0:
      for j in range(n2):
        refined_weights[i][j] /= row_sum
    else:
      # If row_sum is 0, distribute probability equally among valid targets
      if n2 > 0:
        for j in range(n2):
          refined_weights[i][j] = 1.0 / n2
      else:
        for j in range(n2):
          refined_weights[i][j] = 0.0

  # Zero out weights for mappings to non-existent nodes.
  for i in range(n1):
    for j in range(n2, max_node):
      refined_weights[i][j] = 0.0
  for i in range(n1, max_node):
    for j in range(max_node):
      refined_weights[i][j] = 0.0

  return refined_weights