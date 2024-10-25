from abc import ABC, abstractmethod
import networkx as nx

class TspSolver(ABC):
  """
  Abstract base class for TSP solvers.
  """

  @abstractmethod
  def solve(self, graph: nx.Graph):
    """
    Solve the TSP for a given graph.

    Parameters:
    graph (networkx.Graph): The graph representing the TSP problem.
    
    Returns:
    tuple: (tour, total_cost) 
    """
    pass
