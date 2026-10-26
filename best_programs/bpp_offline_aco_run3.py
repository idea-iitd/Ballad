def priority_v2(demand: np.ndarray, capacity: int) -> np.ndarray:
  """
  A highly refined heuristic matrix generation function for the Bin Packing Problem,
  designed to provide superior guidance for Ant Colony Optimization (ACO).
  This version incorporates several advanced features to capture complex item interactions
  and packing efficiencies.

  Key features and improvements in v2:
  1.  **Multi-faceted Fit Scoring:**
      *   **Tight Fit (Gaussian):** Still a core component, rewarding pairs that leave
          minimal waste. `sigma` is tuned for sensitivity.
      *   **"Near-Perfect" Fit Bonus:** Explicitly rewards pairs that fit almost perfectly,
          adding a boost for very efficient packings beyond what the general Gaussian
          might emphasize.
      *   **"Dominant Item Gap" Bonus:** Focuses on scenarios where a large item is paired
          with a smaller item, ensuring the remaining gap is suitable for packing other
          items efficiently. This is distinct from general complementary gap scoring.

  2.  **Advanced Item Pairing Heuristics:**
      *   **"Synergistic Pairing" Bonus:** Specifically rewards pairs of large items
          that fit together very snugly, aiming to 'pre-pack' difficult large items.
      *   **"Complementary Pairing" Bonus:** Rewards pairs of dissimilar sized items that
          leave a "strategic" gap, encouraging diversity in packing. The target gap range
          is more finely tuned.
      *   **Item Size Ratio Logic:** Incorporates the ratio of item sizes to dynamically
          adjust the value of certain pairings. For instance, a good complementary gap
          is more valuable if the items are very different in size.

  3.  **Waste Management and Efficiency Focus:**
      *   **"Waste Minimization" Factor (Multiplicative):** A crucial factor that
          multiplicatively penalizes pairs with significant absolute waste. This
          ensures that even if a pair has good component scores, high waste will
          drastically reduce its heuristic value. It's applied only to non-perfect fits.
      *   **Feasibility Enforcement:** Strict enforcement of capacity constraints, setting
          infeasible pairs to zero.

  4.  **Dynamic Thresholding and Normalization:**
      *   **Dynamic "Dominant Item" Thresholds:** Uses percentiles of demand to adapt
          the definition of "dominant" and "small" items to the specific instance's
          item size distribution.
      *   **Normalized Demand and Capacity Ratios:** Uses normalized demands and ratios
          of remaining capacity to capacity for scoring, making the heuristics more
          robust across different scales of demand and bin capacity.

  5.  **Integrated Scoring and Weighting:**
      *   **Additive and Multiplicative Combinations:** Uses a combination of additive
          bonuses (like synergistic, complementary) on top of a core multiplicative
          heuristic (tight fit * item importance), all then modified by the waste
          minimization factor.
      *   **Tunable Weights:** Weights for different bonus components are provided,
          allowing for fine-tuning of the heuristic's behavior.

  Args:
      demand : np.ndarray
          A 1D NumPy array representing the demand (size) of each item.
          Shape: (n,), where n is the number of items.
      capacity : int
          The maximum capacity of each bin.

  Returns:
      np.ndarray
          A 2D NumPy array representing the heuristic matrix of shape (n, n).
          Each entry `[i, j]` is a score between 0 and 1, indicating the
          suitability of placing item `i` and item `j` into the same bin.
  """
  n = demand.shape[0]
  heuristic_matrix = np.zeros((n, n))

  if n == 0:
    return heuristic_matrix

  # --- Normalization and Basic Calculations ---
  # Normalize demands to [0, 1] for consistent scoring.
  max_demand = np.max(demand)
  if max_demand == 0:
      normalized_demand = np.zeros_like(demand)
  else:
      normalized_demand = demand / max_demand

  # Calculate combined demand and remaining capacity for all pairs.
  combined_demand = demand[:, np.newaxis] + demand[np.newaxis, :]
  remaining_capacity = capacity - combined_demand

  # Create a mask for pairs that exceed bin capacity (infeasible).
  infeasible_mask = combined_demand > capacity

  # --- Component 1: "Tight Fit" Score (Gaussian Decay) ---
  # Rewards pairs that leave minimal space. A smaller sigma leads to a steeper penalty for waste.
  # Tuned sigma for sensitivity to waste. Smaller sigma means stricter requirement for small waste.
  sigma_tight_fit = max(1, capacity * 0.015) # Tighter sigma for even more sensitivity
  tight_fit_score = np.exp(- (remaining_capacity**2) / (2 * sigma_tight_fit**2))
  tight_fit_score[infeasible_mask] = 0 # Infeasible pairs get zero score for this component.

  # --- Component 2: "Item Importance" Score ---
  # Average normalized demand of the pair. Larger items contribute more.
  item_importance_score = (normalized_demand[:, np.newaxis] + normalized_demand[np.newaxis, :]) / 2.0

  # --- Core Heuristic (Multiplicative) ---
  # Combines tight fit and item importance. High score requires both efficiency and significant item sizes.
  core_heuristic = tight_fit_score * item_importance_score

  # --- Bonus Components (Additive) ---

  # Weighting factors for different bonus types. These are tunable.
  w_synergistic = 0.25
  w_complementary = 0.30
  w_dominant_gap = 0.35
  w_near_perfect = 0.10 # New: Bonus for near-perfect fits

  # --- Component 3: "Synergistic Pairing" Bonus ---
  # Rewards pairs of large items that fit together very snugly.
  large_item_threshold = 0.75  # Tunable: defines what is considered a "large" item.

  both_items_large_mask = (normalized_demand[:, np.newaxis] >= large_item_threshold) & \
                          (normalized_demand[np.newaxis, :] >= large_item_threshold)

  # Bonus is applied when both items are large and fit with very little remaining capacity.
  synergy_capacity_limit = capacity * 0.05 # Tunable: maximum allowed waste for this bonus.
  synergy_capacity_mask = (remaining_capacity >= 0) & (remaining_capacity <= synergy_capacity_limit)

  # The bonus is proportional to how tightly they fit.
  synergistic_pairing_bonus = np.where(both_items_large_mask & synergy_capacity_mask, tight_fit_score, 0)

  # --- Component 4: "Complementary Pairing" Bonus ---
  # Rewards pairs of items with different sizes that leave a strategic gap.
  demand_diff_norm = np.abs(normalized_demand[:, np.newaxis] - normalized_demand[np.newaxis, :])

  # Ideal gap range for packing smaller items. Tunable.
  complementary_gap_lower_bound = capacity * 0.15
  complementary_gap_upper_bound = capacity * 0.40

  # Ensure the bounds are valid.
  if complementary_gap_upper_bound <= complementary_gap_lower_bound:
      complementary_gap_upper_bound = complementary_gap_lower_bound + 1.0

  # Score for being within the ideal gap range. Decays linearly.
  gap_range_width = complementary_gap_upper_bound - complementary_gap_lower_bound
  gap_penalty_factor = 1.0 / gap_range_width if gap_range_width > 0 else 1.0

  # Score is 1 at the lower bound and decreases to 0 at the upper bound.
  complementary_gap_score = np.maximum(0.0, 1.0 - gap_penalty_factor * np.maximum(0.0, remaining_capacity - complementary_gap_lower_bound))

  # The bonus is proportional to the demand difference and how well the gap fits the 'strategic' range.
  # Higher demand difference (dissimilar items) paired with a good gap yields a higher bonus.
  complementary_pairing_indicator = demand_diff_norm * complementary_gap_score

  # Apply bonus only to feasible pairs within the specified gap range.
  complementary_mask = (remaining_capacity >= complementary_gap_lower_bound) & \
                       (remaining_capacity <= complementary_gap_upper_bound)
  complementary_gap_bonus = np.where(complementary_mask & ~infeasible_mask, complementary_pairing_indicator, 0)


  # --- Component 5: "Dominant Item Gap" Bonus ---
  # Specifically rewards pairing a large item with a smaller item,
  # ensuring the remaining gap is reasonably sized for packing other items.
  # Dynamic thresholds for "dominant" and "small" items based on demand distribution.
  sorted_demand_norm = np.sort(normalized_demand)
  # If all demands are equal, use fixed thresholds.
  if np.all(normalized_demand == normalized_demand[0]):
      dominant_item_threshold_dynamic = 0.7
      small_item_threshold_dynamic = 0.2
  else:
      dominant_item_threshold_dynamic = np.percentile(sorted_demand_norm, 80) # Top 20%
      small_item_threshold_dynamic = np.percentile(sorted_demand_norm, 30)  # Bottom 30%

  # Ensure thresholds are ordered correctly, or fallback if they overlap significantly.
  if dominant_item_threshold_dynamic < small_item_threshold_dynamic:
      dominant_item_threshold_dynamic, small_item_threshold_dynamic = small_item_threshold_dynamic, dominant_item_threshold_dynamic
  if dominant_item_threshold_dynamic - small_item_threshold_dynamic < 0.15: # Small margin for distinction
      dominant_item_threshold_dynamic = 0.7
      small_item_threshold_dynamic = 0.2

  # Mask for pairs where one item is dominant and the other is small.
  dominant_i_small_j_mask = (normalized_demand[:, np.newaxis] >= dominant_item_threshold_dynamic) & \
                            (normalized_demand[np.newaxis, :] <= small_item_threshold_dynamic)
  dominant_j_small_i_mask = (normalized_demand[np.newaxis, :] >= dominant_item_threshold_dynamic) & \
                            (normalized_demand[:, np.newaxis] <= small_item_threshold_dynamic)
  dominant_pairing_mask = dominant_i_small_j_mask | dominant_j_small_i_mask

  # The bonus is proportional to the demand difference and a normalized remaining capacity.
  # We reward moderate gaps for these pairings. `np.clip(remaining_capacity / capacity, 0, 0.5)`
  # assigns a score from 0 to 0.5 based on how much space is left relative to capacity.
  # A moderate gap is more valuable than a tiny or extremely large one for this bonus type.
  dominant_gap_bonus_score = np.where(dominant_pairing_mask & ~infeasible_mask,
                                      (demand_diff_norm / 2.0) * np.clip(remaining_capacity / capacity, 0.0, 0.5),
                                      0.0)
  dominant_item_gap_bonus = dominant_gap_bonus_score


  # --- Component 6: "Near-Perfect Fit" Bonus ---
  # Rewards pairs with very little remaining capacity, independent of item size.
  near_perfect_capacity_limit = capacity * 0.02 # Max waste for this bonus
  near_perfect_mask = (remaining_capacity >= 0) & (remaining_capacity <= near_perfect_capacity_limit)
  # Bonus is proportional to the tightness of fit. Using tight_fit_score directly here.
  near_perfect_fit_bonus = np.where(near_perfect_mask & ~infeasible_mask, tight_fit_score, 0)


  # --- Multiplicative Waste Minimization Factor ---
  # This factor penalizes pairs that leave significant *absolute* wasted space,
  # unless they are perfectly filled. It's applied *after* additive bonuses.
  absolute_waste = np.maximum(0.0, capacity - combined_demand)
  max_possible_waste = capacity # Max waste for a single item.

  # Normalize waste to a 0-1 scale (0 = no waste, 1 = max waste).
  if max_possible_waste == 0:
      normalized_waste = np.zeros_like(absolute_waste)
  else:
      normalized_waste = absolute_waste / max_possible_waste

  # Exponential decay: exp(-k * normalized_waste). Higher k penalizes waste more.
  waste_penalty_factor = 8.0 # Tunable parameter.
  waste_minimization_factor = np.exp(-waste_penalty_factor * normalized_waste)

  # Perfect fills (remaining capacity close to 0) should not be penalized for waste.
  epsilon = 1e-9 # For floating point comparisons.
  perfect_fill_mask = np.abs(remaining_capacity) < epsilon
  waste_minimization_factor[perfect_fill_mask] = 1.0
  waste_minimization_factor[infeasible_mask] = 0.0 # Infeasible pairs get no score.


  # --- Final Combination of Scores ---

  # Combine core heuristic with weighted additive bonuses.
  combined_additive_bonuses = (w_synergistic * synergistic_pairing_bonus +
                               w_complementary * complementary_gap_bonus +
                               w_dominant_gap * dominant_item_gap_bonus +
                               w_near_perfect * near_perfect_fit_bonus)

  # The final heuristic value is (core + bonuses) * waste_factor.
  # This ensures that efficiency (core) and strategic packing (bonuses) are
  # valued, but significantly penalized if they result in too much waste.
  heuristic_matrix = (core_heuristic + combined_additive_bonuses) * waste_minimization_factor

  # --- Final Processing ---
  # Re-apply infeasibility mask to ensure it's zero.
  heuristic_matrix[infeasible_mask] = 0.0

  # Normalize the entire matrix to the [0, 1] range.
  max_heuristic_value = np.max(heuristic_matrix)
  if max_heuristic_value > 0:
      heuristic_matrix = heuristic_matrix / max_heuristic_value
  else:
      # If all scores are zero (e.g., no feasible pairs), return a zero matrix.
      heuristic_matrix = np.zeros((n, n))

  return heuristic_matrix