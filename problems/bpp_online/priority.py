import numpy as np

def priority_v0(item: float, bins_remain_cap: np.ndarray) -> np.ndarray:
  """
    Computes a priority score for each bin in an Online Bin Packing problem.

    In the Online Bin Packing problem, items arrive sequentially and must be placed
    into bins with limited capacity. This `priority` function assigns a heuristic
    score to each bin, indicating how suitable it is for accommodating the current item.
    The bin with the highest priority score is selected for placement.

    Args:
        item : float
            The size of the current item to be packed.
        bins_remain_cap : np.ndarray
            A 1D array where each element represents the remaining capacity
            of a corresponding bin.

    Returns:
        np.ndarray
            A 1D array of priority scores for all bins. Each score reflects the
            relative suitability of placing the current item in that bin.
            The bin with the highest score is considered the best choice.
  """
  max_bin_cap = max(bins_remain_cap)
  score = (bins_remain_cap - max_bin_cap)**2 / item + bins_remain_cap**2 / (item**2)
  score += bins_remain_cap**2 / item**3
  score[bins_remain_cap > item] = -score[bins_remain_cap > item]
  score[1:] -= score[:-1]
  return score
