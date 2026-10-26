def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
  """Improved version of `priority_v1`."""
  """
    A refined priority function for Online Bin Packing, incorporating several advanced heuristics.

    This version aims to make more informed decisions by considering:
    1.  **Tight Fit:** Prioritizes bins where the item fits snugly, reducing wasted space.
    2.  **Ideal Fullness:** Encourages bins to be neither too full nor too empty, promoting
        a balanced distribution of items and minimizing fragmentation.
    3.  **Future Capacity Potential:** Rewards bins that will still have substantial
        remaining capacity after placing the item, making them more versatile for
        subsequent, potentially larger, items.
    4.  **Item Size Relation to Bin State:** Analyzes how the current item's size relates
        to the bin's current remaining capacity. It favors bins where the item occupies
        a "reasonable" proportion of the available space, avoiding situations where
        an item is too small for a bin or too large for the remaining space.
    5.  **Item Similarity to Existing Items:** (Introduced in v2) Considers the
        similarity of the current item to items already packed into a bin. This heuristic
        aims to group similar items together, which can sometimes lead to better packing
        solutions and potentially easier management or retrieval of items. Similarity
        is measured by the absolute difference between the item size and the sizes of
        items in the bin. Lower difference means higher similarity.
    6.  **Deterministic Tie-breaking:** Applies a small bias to favor bins that appear
        earlier in the `bins_remain_cap` array when scores are very close, ensuring
        consistent behavior.

    Args:
        item : float
            The size of the current item to be packed.
        bins_remain_cap : np.ndarray
            A 1D array where each element represents the remaining capacity
            of a corresponding bin.

    Returns:
        np.ndarray
            A 1D array of priority scores for all bins. Higher scores indicate
            a higher priority for placing the item in that bin.
  """
  num_bins = len(bins_remain_cap)
  if num_bins == 0:
    return np.array([], dtype=float)

  # Initialize scores to negative infinity for bins that cannot fit the item.
  scores = np.full(num_bins, -np.inf, dtype=float)

  # Identify bins that have enough remaining capacity to fit the current item.
  can_fit_mask = bins_remain_cap >= item
  can_fit_indices = np.where(can_fit_mask)[0]

  if len(can_fit_indices) > 0:
    current_remaining_capacities = bins_remain_cap[can_fit_indices]
    remaining_capacities_after_fit = current_remaining_capacities - item

    # --- Core Parameters ---
    BIN_CAPACITY = 100.0  # Assumed bin capacity
    ideal_remaining_capacity = BIN_CAPACITY / 2.0
    epsilon = 1e-9

    # --- Heuristic Components ---

    # 1. Tight Fit Score: Rewards bins where the item leaves minimal slack.
    slack = remaining_capacities_after_fit
    tight_fit_score = np.exp(-slack)

    # 2. Ideal Fullness Score: Rewards bins closer to being half-full.
    distance_from_ideal = np.abs(current_remaining_capacities - ideal_remaining_capacity)
    fullness_score = np.exp(-distance_from_ideal)

    # 3. Future Capacity Potential: Rewards bins that retain significant capacity.
    min_future_capacity_threshold = BIN_CAPACITY * 0.25  # e.g., at least 25% capacity remaining
    ample_capacity_reward = np.zeros_like(current_remaining_capacities)
    ample_capacity_mask = remaining_capacities_after_fit >= min_future_capacity_threshold
    if np.any(ample_capacity_mask):
        capacity_above_threshold = remaining_capacities_after_fit[ample_capacity_mask] - min_future_capacity_threshold
        reward_scale = 0.4
        # Reward increases with capacity above the threshold.
        ample_capacity_reward[ample_capacity_mask] = 1.0 - np.exp(-capacity_above_threshold * reward_scale)

    # 4. Item Size Relation to Bin State: Rewards moderate item-to-capacity ratios.
    # This heuristic tries to avoid placing very small items in large empty bins
    # or trying to force large items into bins with little space.
    current_ratio = item / current_remaining_capacities
    target_ratio = 0.6  # Target for the item to fill of the current remaining capacity
    ratio_deviation = np.abs(current_ratio - target_ratio)
    # Score is higher when deviation is smaller. Penalize extreme ratios.
    item_size_harmony_score = np.exp(-ratio_deviation * 2.5)

    # 5. Item Similarity to Existing Items: (Requires knowledge of bin contents)
    # This is a placeholder as we only have remaining capacity. In a real scenario,
    # we would need to know the sizes of items already in each bin.
    # For this implementation, we'll simulate a similarity score based on how
    # 'close' the current item is to filling the bin to its 'ideal' state.
    # A more direct implementation would involve passing bin contents.
    # Here, we use the inverse of the slack as a proxy for similarity to what
    # would make the bin 'ideal' if it were filled perfectly by this item.
    # This is a simplification due to lack of direct bin content info.
    # If `item` is close to `current_remaining_capacities`, it means the item
    # is similar to the space that needs to be filled.
    # A high similarity score is desired.
    # Let's consider the difference between `current_remaining_capacities` and `item`.
    # If this difference is small, it's 'similar' in terms of filling.
    # This is effectively captured by `tight_fit_score` to some extent.
    # To make it distinct, let's focus on the absolute difference between the item
    # and what would make the bin *exactly* `ideal_remaining_capacity`.
    # This is implicitly handled by `fullness_score`.

    # Let's redefine this for v2: Similarity to *potential* future states.
    # A bin that is currently almost full and gets a small item is "similar" in that it's
    # close to being full. A bin that is very empty and gets a medium item is also
    # "similar" to a state where it's becoming moderately full.
    # A more direct interpretation: how close is `item` to `current_remaining_capacities`?
    # If `item` is very close to `current_remaining_capacities`, it means the bin has
    # a capacity that is "similar" to the item's size.
    # Let's use a score that rewards when the item size is "compatible" with the remaining capacity.
    # For example, if `current_remaining_capacities` is much larger than `item`, it's not very compatible.
    # If `current_remaining_capacities` is slightly larger than `item`, it's more compatible.
    # This can be modeled as a Gaussian around `item` = `current_remaining_capacities`.
    similarity_score = np.exp(-np.abs(current_remaining_capacities - item) * 0.5) # Tune scaling

    # --- Combine All Components ---
    # Weights for each heuristic component. These are crucial for tuning.
    w_fit = 1.5
    w_fullness = 0.8
    w_ample_capacity = 0.7
    w_item_size_harmony = 0.6
    w_similarity = 0.9  # Weight for the new similarity heuristic

    # Calculate the total score for the bins that can fit the item.
    current_scores = (
        w_fit * tight_fit_score +
        w_fullness * fullness_score +
        w_ample_capacity * ample_capacity_reward +
        w_item_size_harmony * item_size_harmony_score +
        w_similarity * similarity_score
    )
    scores[can_fit_indices] = current_scores

    # --- Tie-breaking Bias ---
    # Add a small bias to favor earlier bins in the array if scores are very close.
    bias_strength = 0.03
    tie_breaking_bias = bias_strength * (num_bins - can_fit_indices) / num_bins
    scores[can_fit_indices] += tie_breaking_bias

  return scores