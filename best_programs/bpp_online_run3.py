def priority_v2(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
  """Improved version of `priority_v1`."""
  scores = np.full_like(bins_remain_cap, -np.inf, dtype=float)
  fit_indices = np.where(bins_remain_cap >= item)[0]

  if len(fit_indices) == 0:
    return scores

  current_bins_remain_cap = bins_remain_cap[fit_indices]
  remaining_after_packing = current_bins_remain_cap - item
  scores_fit = 1.0 / (remaining_after_packing + 1e-6)

  # Even stronger penalty for exact fits
  exact_fit_indices = np.where(np.isclose(remaining_after_packing, 0))[0]
  scores_fit[exact_fit_indices] *= 0.7  # Very strong penalty

  # Enhanced bonus for large bins
  large_bin_indices = np.where(current_bins_remain_cap > 2.5 * item)[0]
  scores_fit[large_bin_indices] += 0.2  # Larger bonus

  # Add a small bonus for bins that are close to being full
  almost_full_indices = np.where(current_bins_remain_cap < item * 0.5)[0]
  scores_fit[almost_full_indices] += 0.05  # Small bonus for almost full bins

  scores[fit_indices] = scores_fit

  return scores