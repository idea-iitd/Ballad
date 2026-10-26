def priority_v2(graph1: list[list[int]], graph2: list[list[int]], weights: list[list[int]]) -> list[list[float]]:
  """Improved version of `priority_v1`."""
  n1 = len(graph1)
  n2 = len(graph2)
  max_node = max(n1, n2)

  # Ensure weights matrix has the correct dimensions, padding with zeros if necessary
  padded_weights = [[0.0] * max_node for _ in range(max_node)]
  for r in range(len(weights)):
      for c in range(len(weights[0])):
          if r < max_node and c < max_node:
              padded_weights[r][c] = weights[r][c]
  weights = padded_weights

  refined_weights = [[0.0] * max_node for _ in range(max_node)]

  # Initialize weights based on node degree similarity
  for i in range(n1):
    for j in range(n2):
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])
      similarity = 1.0 - abs(degree1 - degree2) / (max(degree1, degree2) + 1e-6)
      refined_weights[i][j] = similarity * weights[i][j]

  # Consider edge similarity and common neighbors
  for i in range(n1):
    for j in range(n2):
      edge_similarity_count = 0
      common_neighbors_count = 0
      neighbors1 = set()
      neighbors2 = set()

      # Count edges connected to node i in graph1
      for k in range(n1):
          if graph1[i][k] == 1:
              neighbors1.add(k)

      # Count edges connected to node j in graph2
      for l in range(n2):
          if graph2[j][l] == 1:
              neighbors2.add(l)

      # Calculate edge similarity and common neighbors
      for k in neighbors1:
          for l in neighbors2:
              if graph1[i][k] == 1 and graph2[j][l] == 1:
                  edge_similarity_count += 1

      common_neighbors_count = len(neighbors1.intersection(neighbors2))

      # Add edge similarity contribution
      max_degree = max(sum(graph1[i]), sum(graph2[j]))
      if max_degree > 0:
          refined_weights[i][j] += edge_similarity_count / max_degree
      else:
          refined_weights[i][j] += edge_similarity_count # Handle cases with zero degrees

      # Add common neighbors contribution
      total_neighbors = len(neighbors1) + len(neighbors2)
      if total_neighbors > 0:
          refined_weights[i][j] += common_neighbors_count / total_neighbors
      else:
          refined_weights[i][j] += common_neighbors_count # Handle cases with no neighbors

  # Normalize weights to sum to 1 for each row
  for i in range(n1):
    row_sum = sum(refined_weights[i])
    if row_sum > 0:
      for j in range(max_node): # Normalize over max_node to account for potential padding
        refined_weights[i][j] /= row_sum

  # Add a term to penalize mappings between nodes with very different degrees
  degree_diff_penalty = 0.1
  for i in range(n1):
    for j in range(n2):
      degree1 = sum(graph1[i])
      degree2 = sum(graph2[j])
      degree_diff = abs(degree1 - degree2)
      refined_weights[i][j] *= (1 - min(1.0, degree_diff * degree_diff_penalty)) # Cap penalty at 1

  # Add a term to reward mappings where neighbors are also mapped
  neighbor_reward = 0.05
  for i in range(n1):
    for j in range(n2):
      reward = 0.0
      for k in range(n1):
        if graph1[i][k] == 1: # If node i has an edge to node k in graph1
          for l in range(n2):
            if graph2[j][l] == 1: # If node j has an edge to node l in graph2
              reward += refined_weights[k][l]
      refined_weights[i][j] += neighbor_reward * reward

  # Add a term to reward mappings between nodes with similar labels (if labels are available)
  # Assuming labels are the first element of the node's feature vector in the input format.
  # If node features are not provided, this part will be skipped.
  label_similarity_reward = 0.02
  for i in range(n1):
    for j in range(n2):
      # Check if node features are provided and have at least one dimension for label
      if len(graph1[i]) > 0 and len(graph2[j]) > 0:
        # Assuming node features are lists/arrays where the first element is the label.
        # This part needs to be adapted if the input 'nf' for node features has a different structure.
        # For adjacency matrix input, we might need to access node labels separately if they are not encoded in the matrix.
        # For simplicity, let's assume graph1[i][0] and graph2[j][0] are labels if graph[i] has entries.
        # This is a placeholder and might need adjustment based on the actual `nf1` and `nf2` structure.
        pass # Placeholder for label comparison if labels were part of input.

  # Final normalization to ensure probabilities sum to 1 for each row.
  # This step is crucial if previous modifications might have shifted the sums.
  for i in range(n1):
      row_sum = sum(refined_weights[i])
      if row_sum > 0:
          for j in range(max_node):
              refined_weights[i][j] /= row_sum
      else:
          # If a row sum is 0, assign uniform probabilities if there are nodes in graph2
          if n2 > 0:
              for j in range(n2):
                  refined_weights[i][j] = 1.0 / n2
          else:
              for j in range(max_node):
                  refined_weights[i][j] = 0.0


  return refined_weights