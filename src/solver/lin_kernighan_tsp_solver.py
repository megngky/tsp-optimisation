import networkx as nx
import random
from src.solver.tsp_solver import TspSolver

class LinKernighanSolver(TspSolver):
  """
  TSP solver using the Lin-Kernighan heuristic.
  """

  def solve(self, graph: nx.Graph):
    if len(graph.nodes) > 200:
      return None  # Not suitable for large TSP instances
    
    # Step 1: Generate an initial tour using the nearest neighbor heuristic
    initial_tour = self._nearest_neighbor_tour(graph)
    best_tour = initial_tour
    best_cost = self._calculate_tour_cost(graph, best_tour)

    # Step 2: Apply iterative k-opt improvements
    improved = True
    while improved:
      improved = False
      for i in range(len(best_tour) - 1):
        for j in range(i + 2, len(best_tour)):
          # Try a 2-opt move by reversing the segment between i and j
          new_tour = best_tour[:i+1] + best_tour[i+1:j+1][::-1] + best_tour[j+1:]
          new_cost = self._calculate_tour_cost(graph, new_tour)
          if new_cost < best_cost:
            best_tour, best_cost = new_tour, new_cost
            improved = True
            break
        if improved:
          break

    return best_tour, best_cost

  def _nearest_neighbor_tour(self, graph):
    """
    Generate an initial tour using the nearest neighbor heuristic.
    """
    nodes = list(graph.nodes)
    start_node = random.choice(nodes)
    tour = [start_node]
    unvisited = set(nodes)
    unvisited.remove(start_node)

    current_node = start_node
    while unvisited:
      next_node = min(unvisited, key=lambda node: graph[current_node][node]['weight'])
      tour.append(next_node)
      unvisited.remove(next_node)
      current_node = next_node

    # Complete the cycle
    tour.append(start_node)
    return tour

  def _calculate_tour_cost(self, graph, tour, penalty_weight=1e6):
    """
    Calculate the total cost of a given tour.
    """
    total_cost = 0
    for i in range(len(tour) - 1):
      u, v = tour[i], tour[i + 1]
      if graph.has_edge(u, v):
        total_cost += graph[u][v]['weight']
      else:
        total_cost += penalty_weight  # Add a high penalty for missing edges
    return total_cost

