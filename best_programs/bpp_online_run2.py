def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
  """
    Computes a priority score for each bin in an Online Bin Packing problem,
    aiming for a balanced Best Fit strategy that considers both minimizing slack
    and avoiding overly large remaining capacities.

    This version refines the scoring by:
    1.  **Strongly incentivizing near-perfect fits:** Bins that leave minimal
        slack are highly prioritized.
    2.  **Penalizing large slacks more aggressively:** The penalty for large
        remaining capacity increases with the slack.
    3.  **Introducing a "bin fullness" heuristic (implicit):** While original
        bin capacities aren't known, we can infer that bins with less remaining
        capacity (that still fit the item) were likely fuller or have had more
        items packed into them. This implicitly favors bins that are already
        somewhat "used".
    4.  **Smooth transition between slack categories:** The scoring function
        aims for a smoother transition between different slack levels to avoid
        sudden jumps in priority.

    Scoring Logic:
    - **Perfect Fit:** Highest possible score.
    - **Near Perfect Fit:** Very high scores, decreasing slowly as slack increases.
      This rewards bins that are "almost full".
    - **Good Fit (Moderate Slack):** A score that decreases more noticeably with slack.
    - **Large Slack (Penalized):** A significantly steeper decrease in score as slack grows.
    - **Cannot Fit:** -np.inf.

    Key parameters:
    - `epsilon`: For identifying perfect fits.
    - `near_perfect_slack_threshold`: Defines the upper bound for "near perfect" fits.
    - `aggressive_penalty_start_ratio`: The point at which slack becomes significantly large relative to item size.
    - `penalty_multiplier`: Controls how aggressively scores drop for large slacks.
    - `slack_decay_power`: Exponent to control the rate of score decay for larger slacks.

    Args:
        item : float
            The size of the current item to be packed.
        bins_remain_cap : np.ndarray
            A 1D array where each element represents the remaining capacity
            of a corresponding bin.

    Returns:
        np.ndarray
            A 1D array of priority scores for all bins. The bin with the
            highest score is considered the best choice.
  """
  # Initialize scores to negative infinity for bins that cannot fit the item.
  scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)

  # Identify bins that have enough remaining capacity for the current item.
  can_fit_indices = np.where(bins_remain_cap >= item)[0]

  if len(can_fit_indices) > 0:
    # Calculate the slack for bins that can accommodate the item.
    # Slack = Remaining Capacity - Item Size
    slack = bins_remain_cap[can_fit_indices] - item

    # --- Scoring Parameters ---
    # Threshold for perfect fits (extremely small slack)
    epsilon = 1e-9

    # Threshold for "near-perfect" fits (e.g., slack is up to 10% of item size)
    near_perfect_slack_threshold = item * 0.1

    # Ratio of slack to item size beyond which we start applying a more aggressive penalty.
    aggressive_penalty_start_ratio = 0.6

    # Multiplier for the penalty function, controls the steepness of the score drop.
    # A higher value means a more aggressive penalty.
    penalty_multiplier = 25.0

    # Power for the slack decay function. Higher values lead to faster decay for larger slacks.
    slack_decay_power = 2.5

    # Base score for "good" fits. This score will be modulated by the slack ratio.
    good_fit_base_score = 0.9

    # --- Scoring Logic ---
    for i, idx in enumerate(can_fit_indices):
      current_slack = slack[i]

      if current_slack < epsilon:
        # Perfect Fit: Assign the absolute highest score.
        scores[idx] = 1e12
      elif current_slack <= near_perfect_slack_threshold:
        # Near-Perfect Fit: High score, decreasing very slowly with slack.
        # This prioritizes bins that are "almost full" but still accommodate the item.
        # The formula is designed to give high scores but with a gentle decay.
        scores[idx] = 1e10 - current_slack * 1e8
      else:
        # For larger slacks, we use a more dynamic scoring function.
        slack_ratio = current_slack / item

        if slack_ratio < aggressive_penalty_start_ratio:
          # Good Fit (Moderate Slack): Score decreases gradually.
          # The score is based on a good_fit_base_score and decays inverse-proportionally
          # to (1 + slack_ratio).
          scores[idx] = good_fit_base_score * (1.0 / (1 + slack_ratio))
        else:
          # Large Slack (Aggressively Penalized): Score drops sharply.
          # This function heavily penalizes bins with significantly more space than needed.
          # The `slack_decay_power` controls how quickly the score diminishes.
          scores[idx] = penalty_multiplier * (1.0 / (1 + slack_ratio**slack_decay_power))

        # Ensure scores remain positive and do not go below a small threshold.
        scores[idx] = max(scores[idx], 1e-8)

    # Final check to ensure perfect fits are indeed the highest, overriding any
    # potential floating-point issues that might have made a near-perfect fit score higher.
    perfect_fit_mask = (slack < epsilon)
    if np.any(perfect_fit_mask):
        scores[can_fit_indices[perfect_fit_mask]] = 1e12

  return scores