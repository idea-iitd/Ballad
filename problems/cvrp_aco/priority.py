
import numpy as np
from copy import copy
from scipy.spatial import distance_matrix


def priority_v0(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
  """
  Computes prior indicators of how promising it is to include each edge in a 
  solution for the Capacitated Vehicle Routing Problem (CVRP).
  
  The CVRP extends TSP by using multiple vehicles with limited capacity to serve
  customers with known demands, minimizing total travel distance while respecting
  capacity constraints. The depot node is indexed by 0 and serves as the starting
  and ending point for all vehicle routes.
  
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
  
  return 1 / distance_matrix