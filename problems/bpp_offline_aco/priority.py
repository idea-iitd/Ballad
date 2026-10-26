import numpy as np

def priority_v0(demand: np.ndarray, capacity: int) -> np.ndarray:
  """
  Computes a heuristic matrix that represents the relative suitability of 
  grouping items together in the same bin for a bin packing problem.

  The output is a square matrix where each entry [i, j] indicates how 
  promising it is to place item i and item j in the same bin, considering 
  their demand values. Pairs (i, j) whose combined demand 
  exceeds the bin capacity should be assigned a lower heuristic score, 
  as they cannot feasibly be placed in the same bin.

  Args:
      demand : np.ndarray
          1D array of demand values for n items (shape: (n,)). 
      capacity : int
          Integer specifying the maximum capacity of each bin. This provides 
          an upper limit that should guide which item pairs are practically 
          feasible to be grouped together.

  Returns:
      np.ndarray
          2D heuristic matrix of shape (n, n). 
          Entry [i, j] signifies the prior indicator of how suitable it is 
          to assign item i and item j to the same bin, taking demand 
          compatibility and capacity feasibility into account.
  """
  return np.tile(demand/demand.max(), (demand.shape[0], 1))